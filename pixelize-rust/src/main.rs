#![allow(non_snake_case)]

pub mod util;
pub mod pixelize;
pub mod denoise;
pub mod model_data;

use std::{env, io::Cursor};

use image::{io::Reader as ImageReader, Rgb32FImage};
use image;

use std::collections::HashMap;
use onnxruntime::{
    environment::Environment, 
    ndarray::{Array, Ix4},
    LoggingLevel
};


fn process_image_all(models: &mut HashMap<String, onnxruntime::session::Session<'_>>, image_path: Vec<String>) -> Result<(),()>{
    let reference_b = Cursor::new(include_bytes!("../../reference.png"));
    let reference_image = ImageReader::new(reference_b).with_guessed_format().unwrap().decode().unwrap().into_rgb32f();
    let reference_arr = pixelize::normalize(pixelize::grayscale(util::image_to_arr(reference_image, 256, 256)));
    for filename in image_path{
        // load
        println!("{}", filename);
        let media = util::load_file(filename.clone());
        let image = match media{
            util::Media::Frames((images, bgcolor)) => Rgb32FImage::new(1,1),
            util::Media::Frame((image, bgcolor)) => image.into_rgb32f(),
        };

        // transform
        let mut data = Rgb32FImage::from_pixel(1000, 1000, image::Rgb::<f32>([0.5_f32,0.5_f32,0.5_f32]));
        image::imageops::overlay(&mut data, &image, 0, 0);

        // pixelize
        let data_arr = pixelize::normalize(util::image_to_arr(data, 1000, 1000));
        let output: Array<f32, Ix4> = pixelize::process_image(models, data_arr, reference_arr.clone());
        let mut img_p = util::arr_to_image(output, 1000, 1000);

        // save
        let sub_img_p = image::imageops::crop(&mut img_p, 0, 0, image.width(), image.height());
        let path = format!("{}.pixelized.png", std::path::Path::new(&filename).file_stem().unwrap().to_str().unwrap());
        sub_img_p.to_image().save_with_format(&path, image::ImageFormat::Png).expect("File save failed");
    }
    Ok(())
}

fn main(){
    let args: Vec<String> = env::args().collect();
    if args.len() == 1{
        println!("Help message:\n\nUsage:\npixelize.exe img1 img2 ...\n\nOriginal model: Zongwei Wu (github.com/WuZongWei6/Pixelization)\nPackaging: Jongkook Choi (github.com/Thessal/pixelization_inference)\n");
        ()
    }

    let environment = Environment::builder()
        .with_name("test")
        .with_log_level(LoggingLevel::Info)
        .build().unwrap();
    let mut models = model_data::load_model_all(&environment);

    process_image_all(&mut models, args[1..].to_vec()).expect("Main loop failed");
}
