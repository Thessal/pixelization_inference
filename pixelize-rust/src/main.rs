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
    ndarray::{Array, Ix4, Ix5, concatenate, Axis},
    LoggingLevel
};


fn process_image_all(models: &mut HashMap<String, onnxruntime::session::Session<'_>>, image_path: Vec<String>) -> Result<(),()>{
    let noise_level = 0.1_f32;
    let size_h:usize = 1000;
    let size_w:usize = 1000;

    let images_all: Vec<(String, Rgb32FImage)> = image_path.iter().map(
        |filename| {
            let media = util::load_file( filename.clone() );
            let image = match media{
                util::Media::Frames((images, bgcolor)) => vec![(filename.clone(), Rgb32FImage::new(1,1))],
                util::Media::Frame((image, bgcolor)) => vec![(filename.clone(), image.into_rgb32f())],
            };
            image
        }).flatten().collect();

    // transform
    let mut arrs_all: Vec<(String, Array<f32, Ix4>, (u32, u32))> = images_all.iter().map(
        |filename_image| {
            let (filename, image) = filename_image;
            let mut data = Rgb32FImage::from_pixel(
                size_h as u32, size_w as u32, image::Rgb::<f32>([0.5_f32,0.5_f32,0.5_f32]));
            image::imageops::overlay(&mut data, image, 0, 0);
            (filename.clone(), util::image_to_arr(data, size_h, size_w), (image.height(), image.width()))
        }).collect();

    //denoise
    let n_frames = images_all.len();
    if n_frames >= 5 {
        let mut denoised_all: Vec<(String, Array<f32, Ix4>, (u32,u32))> = Vec::new();
        for i in 0..n_frames{
            // FIXME
            println!("Denoising {}", &arrs_all[i].0);
            let f0 = &arrs_all[std::cmp::max(i-2, 0)].1;
            let f1 = &arrs_all[std::cmp::max(i-1, 0)].1;
            let f2 = &arrs_all[i].1;
            let f3 = &arrs_all[std::cmp::min(n_frames-1, i+1)].1;
            let f4 = &arrs_all[std::cmp::min(n_frames-1, i+2)].1;
            let frames : Array<f32,Ix4> = ndarray::concatenate![Axis(0), f0.clone(), f1.clone(), f2.clone(), f3.clone(), f4.clone()]; 
            let frames : Array<f32,Ix5> = frames.insert_axis(Axis(0));
            let denoised = denoise::denoise(models, frames, noise_level);
            denoised_all.push( (arrs_all[i].0.clone(), denoised, arrs_all[i].2) );
        }
        arrs_all = denoised_all;
    }
    
    //pixelize
    let reference_b = Cursor::new(include_bytes!("../../reference.png"));
    let reference_image = ImageReader::new(reference_b).with_guessed_format().unwrap().decode().unwrap().into_rgb32f();
    let reference_arr = pixelize::normalize(pixelize::grayscale(util::image_to_arr(reference_image, 256, 256)));
    for (filename, data, (img_h, img_w)) in arrs_all{
        // pixelize
        println!("Pixelizing {}", filename);
        let data_arr = pixelize::normalize(data);
        let output: Array<f32, Ix4> = pixelize::process_image(models, data_arr, reference_arr.clone());
        let mut img_p = util::arr_to_image(output, 1000, 1000);

        //save
        let sub_img_p = image::imageops::crop(&mut img_p, 0, 0, img_h, img_w);
        let path = format!("{}.pixelized.png", std::path::Path::new(&filename).file_stem().unwrap().to_str().unwrap());
        sub_img_p.to_image().save_with_format(&path, image::ImageFormat::Png).expect("File save failed");
    }

    // TODO : pallete builder
    Ok(())
}

fn main(){
    let args: Vec<String> = env::args().collect();
    if args.len() == 1{
        println!("Help message:\n\nUsage:\npixelize.exe img1 img2 ...\n\nPixelization: Zongwei Wu (github.com/WuZongWei6/Pixelization)\nDenoise: Matias Tassano (https://github.com/m-tassano/fastdvdnet)\nPackaging: Jongkook Choi (github.com/Thessal/pixelization_inference)\n");
        ()
    }

    let environment = Environment::builder()
        .with_name("test")
        .with_log_level(LoggingLevel::Info)
        .build().unwrap();
    let mut models = model_data::load_model_all(&environment);

    process_image_all(&mut models, args[1..].to_vec()).expect("Main loop failed");
}
