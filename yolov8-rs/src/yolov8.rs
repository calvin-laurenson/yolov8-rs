use super::{BoundingBox, Config};
use image::imageops;
use image::{imageops::FilterType, DynamicImage, GenericImageView};
use ndarray::CowArray;
use ndarray::{s, Array, Axis, IxDyn};
use ort::Value;
use ort::{Environment, SessionBuilder};
use std::collections::HashMap;
use std::num::ParseIntError;
use std::{path::Path, sync::Arc, vec};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ClassesParseError {
    #[error("End of string reached early")]
    EarlyEnd,
    #[error("Invalid number")]
    InvalidNumber(#[from] ParseIntError),
}

#[derive(Error, Debug)]
pub enum InitializationError {
    #[error("Internal error in the ort library")]
    Ort(#[from] ort::OrtError),
    #[error("Classes metadata not found")]
    ClassesMissing,
    #[error("Error parsing classes")]
    ClassesParse(#[from] ClassesParseError),
    #[error("Invalid model")]
    InvalidModel,
}

#[derive(Error, Debug)]
pub enum DetectionError {
    #[error("Internal error in the ort library")]
    Ort(#[from] ort::OrtError),
    #[error("Error preprocessing input")]
    PreprocessingInput(#[from] PreprocessImagesError),
    #[error("Error processing input")]
    ProcessingInput(#[from] ProcessInputError),
    #[error("Error processing output")]
    ProcessingOutput(#[from] ProcessOutputError),
    #[error("Error postprocessing output")]
    PostprocessingOutput(#[from] PostprocessOutputError),
}

#[derive(Error, Debug)]
pub enum PreprocessDynamicImagesError {
    #[error("No images provided")]
    NoImages,
    #[error("Image sizes mismatch")]
    ImageSizeMismatch,
}

#[derive(Error, Debug)]
pub enum PreprocessStaticImagesError {}

#[derive(Error, Debug)]
pub enum PreprocessImagesError {
    #[error("Error preprocessing images for dynamic model")]
    Dynamic(#[from] PreprocessDynamicImagesError),
    #[error("Error preprocessing images for static size model")]
    Static(#[from] PreprocessStaticImagesError),
}

#[derive(Error, Debug)]
pub enum ProcessInputError {}

#[derive(Error, Debug)]
pub enum ProcessOutputError {
    #[error("Invalid output format")]
    InvalidFormat,
    #[error("Invalid class id")]
    InvalidClassId,
}

#[derive(Error, Debug)]
pub enum PostprocessOutputError {}

/// Converts the names metadata from the model to a HashMap
/// The names metadata looks like: `{0: 'person', 1: 'bicycle', 2: 'car'}`
fn parse_yolov8_names(input: String) -> Result<HashMap<i32, String>, ClassesParseError> {
    let mut result = HashMap::new();
    let input = input.replace("{", "").replace("}", "");
    for class in input.split(", ") {
        let mut class = class.split(": ");
        let index = class
            .next()
            .ok_or(ClassesParseError::EarlyEnd)?
            .parse::<i32>()
            .map_err(ClassesParseError::InvalidNumber)?;
        let name = class
            .next()
            .ok_or(ClassesParseError::EarlyEnd)?
            .replace("'", "");
        result.insert(index, name);
    }
    Ok(result)
}

/// Adds padding to an image so that its dimensions are a multiple of 32
fn round_image(image: &DynamicImage, round: u32) -> DynamicImage {
    let (width, height) = image.dimensions();
    // Convert to i32
    let width = width as u32;
    let height = height as u32;
    let new_width = width + (round - width % round);
    let new_height = height + (round - height % round);
    let mut new_image = DynamicImage::new_rgb8(new_width, new_height);
    imageops::overlay(&mut new_image, image, 0, 0);
    new_image
}

/// Represents a YOLOv8 model
pub struct YOLOV8 {
    session: ort::Session,
    object_names: HashMap<i32, String>,
    dynamic_batch_size: bool,
    image_size: Option<(u32, u32)>,
}

impl YOLOV8 {
    /// Initializes a YOLOv8 model from an ONNX model file
    pub fn new(
        environment: &Arc<Environment>,
        model_path: &str,
    ) -> Result<Self, InitializationError> {
        let model = SessionBuilder::new(environment)
            .map_err(InitializationError::Ort)?
            .with_model_from_file(Path::new(model_path))
            .map_err(InitializationError::Ort)?;

        let object_names = parse_yolov8_names(
            model
                .metadata()
                .map_err(|_| InitializationError::ClassesMissing)?
                .custom("names")
                .map_err(|_| InitializationError::ClassesMissing)?
                .ok_or(InitializationError::ClassesMissing)?,
        )
        .map_err(InitializationError::ClassesParse)?;
        let input_dims: Vec<Option<usize>> = model
            .inputs
            .get(0)
            .ok_or(InitializationError::InvalidModel)?
            .dimensions()
            .collect();

        let dynamic_batch_size = input_dims[0] == None;
        let image_size = match (input_dims[2], input_dims[3]) {
            (Some(width), Some(height)) => Some((height as u32, width as u32)),
            _ => None,
        };

        if model.outputs.len() < 1 {
            return Err(InitializationError::InvalidModel);
        }

        Ok(YOLOV8 {
            session: model,
            object_names,
            dynamic_batch_size,
            image_size,
        })
    }

    /// Runs the model on multiple images
    pub fn run_batch(
        &self,
        images: Vec<DynamicImage>,
        config: &Config,
    ) -> Result<Vec<(DynamicImage, Vec<BoundingBox>)>, DetectionError> {
        let original_image_sizes = &images
            .iter()
            .map(|image| image.dimensions())
            .collect::<Vec<_>>();
        if !self.dynamic_batch_size && images.len() > 1 {
            // Run the model on each image individually
            return Ok(images
                .into_iter()
                .map(
                    |image| -> Result<(DynamicImage, Vec<BoundingBox>), DetectionError> {
                        Ok(self.run_batch(vec![image], config)?.pop().unwrap()) // This unwrap is ok because we know that there is one going in so one should always come out
                    },
                )
                .collect::<Result<Vec<_>, DetectionError>>()?);
        }
        let (resized_images, image_size) = self
            .preprocess_images(&images, config)
            .map_err(DetectionError::PreprocessingInput)?;
        // Loop over images and create an ndarray with all processed images
        let input = self
            .process_input(&resized_images, image_size)
            .map_err(DetectionError::ProcessingInput)?;
        // Create an ndarray with all images

        let input_tensor =
            Value::from_array(self.session.allocator(), &input).map_err(DetectionError::Ort)?;
        // eprintln!("{:#?}", input_tensor.shape());
        let outputs = self
            .session
            .run(vec![input_tensor])
            .map_err(DetectionError::Ort)?;
        let output = outputs
            .get(0)
            .unwrap() // Should be ok because we checked if there was at least one output
            .try_extract::<f32>()
            .map_err(DetectionError::Ort)?
            .view()
            .t()
            .into_owned();
        let boxes = self
            .process_output(output)
            .map_err(DetectionError::ProcessingOutput)?;
        let filtered_boxes = self
            .postprocess_output(boxes, &config)
            .map_err(DetectionError::PostprocessingOutput)?;
        if config.undo_resize {
            let shifted_boxes =
                if self.image_size.is_none() && config.resize_type == super::ResizeType::Pad {
                    // If any of the boxes are inside of the padded area, shift them
                    // No need to scale the boxes
                    filtered_boxes
                        .into_iter()
                        .zip(original_image_sizes.into_iter())
                        .map(|(boxes, size)| {
                            boxes
                                .into_iter()
                                .map(|box_| BoundingBox {
                                    x2: box_.x2 - (size.0 - image_size.0) as f32,
                                    y2: box_.y2 - (size.1 - image_size.1) as f32,
                                    ..box_
                                })
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>()
                } else {
                    filtered_boxes
                        .into_iter()
                        .zip(original_image_sizes.into_iter())
                        .map(|(boxes, size)| {
                            boxes
                                .into_iter()
                                .map(|mut box_| {
                                    let x_scale = size.0 as f32 / image_size.0 as f32;
                                    let y_scale = size.1 as f32 / image_size.1 as f32;
                                    box_.x1 *= x_scale;
                                    box_.x2 *= x_scale;
                                    box_.y1 *= y_scale;
                                    box_.y2 *= y_scale;
                                    box_
                                })
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>()
                };

            Ok(images
                .into_iter()
                .zip(shifted_boxes.into_iter())
                .collect::<Vec<_>>())
        } else {
            Ok(resized_images
                .into_iter()
                .zip(filtered_boxes.into_iter())
                .collect::<Vec<_>>())
        }
    }

    /// Run the model on a single image
    pub fn run(
        &self,
        image: DynamicImage,
        config: &Config,
    ) -> Result<Vec<BoundingBox>, DetectionError> {
        Ok(self.run_batch(vec![image], config)?.pop().unwrap().1) // Unwrap is ok because we put one in and should always get one out
    }

    fn preprocess_images(
        &self,
        images: &Vec<DynamicImage>,
        config: &Config,
    ) -> Result<(Vec<DynamicImage>, (u32, u32)), PreprocessImagesError> {
        match self.image_size {
            Some(_) => self
                .preprocess_static_images(images, config)
                .map_err(PreprocessImagesError::Static),
            None => self
                .preprocess_dynamic_images(images, config)
                .map_err(PreprocessImagesError::Dynamic),
        }
    }
    fn preprocess_dynamic_images(
        &self,
        images: &Vec<DynamicImage>,
        config: &Config,
    ) -> Result<(Vec<DynamicImage>, (u32, u32)), PreprocessDynamicImagesError> {
        if images.len() == 0 {
            return Err(PreprocessDynamicImagesError::NoImages);
        }

        let mut image_size: Option<(u32, u32)> = None;

        for image in images {
            if let Some((width, height)) = image_size {
                if image.width() != width || image.height() != height {
                    return Err(PreprocessDynamicImagesError::ImageSizeMismatch);
                }
            } else {
                image_size = Some((image.width(), image.height()));
            }
        }

        let image_size = image_size.unwrap(); // This unwrap is ok because the for loop has to run at least once which means this variable has to have a value

        // If the image size is not a multiple of 32, add padding

        let resized_images = if image_size.0 % 32 != 0 || image_size.1 % 32 != 0 {
            match config.resize_type {
                super::ResizeType::Pad => images
                    .into_iter()
                    .map(|image| round_image(image, 32))
                    .collect::<Vec<_>>(),
                super::ResizeType::Resize(filter_type) => images
                    .into_iter()
                    .map(|image| {
                        image.resize_exact(
                            ((image_size.0 as f32 / 32.0).ceil() * 32.0) as u32,
                            ((image_size.1 as f32 / 32.0).ceil() * 32.0) as u32,
                            filter_type,
                        )
                    })
                    .collect::<Vec<_>>(),
            }
        } else {
            images.clone()
        };
        let image_size = resized_images.get(0).unwrap().dimensions(); // This unwrap is ok because we checked to make sure we have at least one image already
        Ok((resized_images, image_size))
    }
    fn preprocess_static_images(
        &self,
        images: &Vec<DynamicImage>,
        config: &Config,
    ) -> Result<(Vec<DynamicImage>, (u32, u32)), PreprocessStaticImagesError> {
        let image_size = self.image_size.unwrap(); // This unwrap is ok because for this function to be called it has to check if image_size is none
        let filter_type = match config.resize_type {
            super::ResizeType::Resize(filter_type) => filter_type,
            _ => FilterType::Nearest,
        };

        Ok((
            images
                .into_iter()
                .map(|image| image.resize_exact(image_size.0, image_size.1, filter_type))
                .collect::<Vec<_>>(),
            image_size,
        ))
    }
    fn process_input(
        &self,
        images: &Vec<DynamicImage>,
        image_size: (u32, u32),
    ) -> Result<CowArray<f32, IxDyn>, ProcessInputError> {
        let mut input = Array::zeros((
            images.len(),
            3,
            image_size.1 as usize,
            image_size.0 as usize,
        ))
        .into_dyn();
        for (i, image) in images.iter().enumerate() {
            for pixel in image.pixels() {
                let (x, y) = (pixel.0 as usize, pixel.1 as usize);
                let rgb = pixel.2;
                input[[i, 0, y as usize, x as usize]] = rgb[0] as f32 / 255.0;
                input[[i, 1, y as usize, x as usize]] = rgb[1] as f32 / 255.0;
                input[[i, 2, y as usize, x as usize]] = rgb[2] as f32 / 255.0;
            }
        }
        Ok(CowArray::from(input))
    }
    /// Process the output of the model into a vector of bounding boxes
    /// # Arguments
    /// * `output` - A 3D array with the shape [num_boxes, (4 + num_classes), batch_size]
    fn process_output(
        &self,
        output: Array<f32, IxDyn>,
    ) -> Result<Vec<Vec<BoundingBox>>, ProcessOutputError> {
        let batch_size = output.shape()[2];
        let mut boxes: Vec<Vec<BoundingBox>> = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let mut batch_boxes = Vec::new();
            let image_output = output.slice(s![.., .., i]);
            for row in image_output.axis_iter(Axis(0)) {
                // The row format is [x_center, y_center, w, h, class_0_prob, class_1_prob, ...]
                let row: Vec<_> = row.iter().map(|x| *x).collect();
                let (class_id, prob) = row
                    .iter()
                    // Skip the first 4 elements because they are the bounding box coordinates
                    .skip(4)
                    // Enumerate the remaining elements to essentially zip the class id with the probability
                    .enumerate()
                    // .map(|(index, value)| (index, *value))
                    // Reduce operation to find the element with the highest probability
                    .reduce(|highest_class, class| {
                        if class.1 > highest_class.1 {
                            class
                        } else {
                            highest_class
                        }
                    })
                    .ok_or(ProcessOutputError::InvalidFormat)?;
                let label = self
                    .object_names
                    .get(&(class_id as i32))
                    .ok_or(ProcessOutputError::InvalidClassId)?;
                let xc = row.get(0).ok_or(ProcessOutputError::InvalidFormat)?;
                let yc = row.get(1).ok_or(ProcessOutputError::InvalidFormat)?;
                let w = row.get(2).ok_or(ProcessOutputError::InvalidFormat)?;
                let h = row.get(3).ok_or(ProcessOutputError::InvalidFormat)?;
                let x1 = xc - w / 2.0;
                let x2 = xc + w / 2.0;
                let y1 = yc - h / 2.0;
                let y2 = yc + h / 2.0;
                batch_boxes.push(BoundingBox {
                    class: label.clone(),
                    confidence: *prob,
                    x1,
                    x2,
                    y1,
                    y2,
                });
            }
            boxes.push(batch_boxes);
        }
        Ok(boxes)
    }

    fn postprocess_output(
        &self,
        boxes: Vec<Vec<BoundingBox>>,
        config: &Config,
    ) -> Result<Vec<Vec<BoundingBox>>, PostprocessOutputError> {
        let mut boxes = boxes
            .into_iter()
            .map(|batch| {
                batch
                    .into_iter()
                    .filter(|box_| box_.confidence > config.thresholds.confidence)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        for batch in boxes.iter_mut() {
            batch.sort_by(|a, b| {
                a.confidence
                    .partial_cmp(&b.confidence)
                    .unwrap_or(std::cmp::Ordering::Greater)
                    .reverse()
            });
        }
        Ok(boxes
            .into_iter()
            .map(|batch| {
                let mut batch = batch;
                let mut i = 0;
                while i < batch.len() {
                    let mut j = i + 1;
                    while j < batch.len() {
                        if batch[i].iou(&batch[j]) > config.thresholds.iou {
                            batch.remove(j);
                        } else {
                            j += 1;
                        }
                    }
                    i += 1;
                }
                batch
            })
            .collect::<Vec<_>>())
    }
}
