use image::imageops::FilterType;
pub use yolov8::YOLOV8;
mod yolov8;

#[derive(Debug, Clone, PartialEq)]
pub struct BoundingBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub class: String,
    pub confidence: f32,
}

impl BoundingBox {
    pub fn area(&self) -> f32 {
        (self.x2 - self.x1) * (self.y2 - self.y1)
    }

    pub fn union(&self, other: &BoundingBox) -> f32 {
        self.area() + other.area() - self.intersection(other)
    }

    pub fn intersection(&self, other: &BoundingBox) -> f32 {
        let x1 = self.x1.max(other.x1);
        let y1 = self.y1.max(other.y1);
        let x2 = self.x2.min(other.x2);
        let y2 = self.y2.min(other.y2);
        (x2 - x1) * (y2 - y1)
    }

    pub fn iou(&self, other: &BoundingBox) -> f32 {
        self.intersection(other) / self.union(other)
    }
}
#[derive(Debug, Clone, PartialEq)]
pub struct Thresholds {
    /// Intersection over union threshold
    pub iou: f32,
    /// Confidence threshold
    pub confidence: f32,
}
#[derive(Debug, Clone, PartialEq)]
pub enum ResizeType {
    /// Add black padding to the image
    Pad,
    /// Resize the image
    Resize(FilterType),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    /// Object detection result thresholds
    pub thresholds: Thresholds,
    /// Method of resizing the image
    pub resize_type: ResizeType,
    /// Undo the resizing of the image after detection
    pub undo_resize: bool,
}
