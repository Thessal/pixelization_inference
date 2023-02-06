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
    ndarray::{Array, Ix4, Ix5, concatenate, Axis, s},
    LoggingLevel
};


fn process_image_all(models: &mut HashMap<String, onnxruntime::session::Session<'_>>, image_path: Vec<String>) -> Result<(),()>{
    let noise_level = 0.1_f32;
    let size_h:usize = 1024;
    let size_w:usize = 1024;

    let images_all: Vec<(String, Rgb32FImage)> = image_path.iter().map(
        |filename| {
            let media = util::load_file( filename.clone() );
            let image = match media{
                util::Media::Frames((images, bgcolor)) => vec![(filename.clone(), images[0].0.clone().into_rgb32f())], //FIXME
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

    //denoise before pixelize
    arrs_all = denoise::denoise_1024(models, arrs_all, noise_level);

    //pixelize
    let reference_b = Cursor::new(include_bytes!("../../reference.png"));
    let reference_image = ImageReader::new(reference_b).with_guessed_format().unwrap().decode().unwrap().into_rgb32f();
    let reference_arr = pixelize::normalize(pixelize::grayscale(util::image_to_arr(reference_image, 256, 256)));
    let mut pixelized_all: Vec<(String, Array<f32, Ix4>, (u32,u32))> = Vec::new();
    for (filename, data, (img_h, img_w)) in arrs_all{
        // pixelize
        println!("Pixelizing {}", filename);
        let data_arr = pixelize::normalize(data);
        let pixel_4x_arr_: Array<f32, Ix4> = pixelize::process_image(models, data_arr, reference_arr.clone());
        let pixel_4x_arr = pixelize::denormalize(pixel_4x_arr_);

        // resize 1/4
        //let pixel_1x_arr = util::arr_quarter(pixel_4x_arr, size_h as u32, size_w as u32);
        //let pixel_1x_arr = pixel_4x_arr.slice(s![..,..,..;4,..;4]).as_standard_layout().to_owned();
        let pixel_1x_arr = 0.25 * (
            pixel_4x_arr.slice(s![..,..,..;4,..;4]).as_standard_layout().to_owned()
            +pixel_4x_arr.slice(s![..,..,1..;4,..;4]).as_standard_layout().to_owned()
            +pixel_4x_arr.slice(s![..,..,..;4,1..;4]).as_standard_layout().to_owned()
            +pixel_4x_arr.slice(s![..,..,1..;4,1..;4]).as_standard_layout().to_owned()
            );
        
        pixelized_all.push( (filename, pixel_1x_arr, (img_h, img_w)) );
    }

    //denoise after pixelize
    let pixelized_denoised_all = denoise::denoise_256(models, pixelized_all, noise_level);

    for (filename, data_arr, (img_h, img_w)) in pixelized_denoised_all.iter(){
        //resize
        let data_arr_ = data_arr.to_owned();
        let img_p_ = util::arr_to_image(data_arr_, size_h as u32 /4 , size_w as u32/4);
        let mut img_p = image::imageops::resize(&img_p_, size_h as u32, size_w as u32, image::imageops::FilterType::Nearest);

        //save
        let sub_img_p = image::imageops::crop(&mut img_p, 0, 0, *img_w, *img_h);
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
