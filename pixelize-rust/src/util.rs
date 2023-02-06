#![allow(non_snake_case)]
use image::{io::Reader as ImageReader, Rgb32FImage, RgbImage, AnimationDecoder, ImageDecoder};
use image;
use std::fs::File;

use onnxruntime::{
    ndarray::{Array, Ix3, Ix4, s},
};

pub fn image_to_arr(image: Rgb32FImage, h:usize, w:usize) -> Array<f32,Ix4> {
    let mut arr = Array::from_iter(image.iter().map(|x| *x)).into_shape((1, h, w, 3)).unwrap();
    arr.swap_axes(1, 3); //hwc->cwh
    arr.swap_axes(2, 3); //cwh->chw
    arr = arr.as_standard_layout().to_owned();
    //arr *= 255.0_f32;
    arr
}

//pub fn arr_quarter(arr:Array<f32, Ix4>, h:u32, w:u32) -> Array<f32, Ix4>{
//    let mut arr_f : Array<f32, Ix3> = arr.slice(s![0,..,..,..]).to_owned();
//    arr_f.swap_axes(1,2); //cxy->cyx
//    arr_f.swap_axes(0,2); //cyx->xyc
//    arr_f = arr_f.as_standard_layout().to_owned();
//    let mut image = Rgb32FImage::from_raw(h, w, arr_f.into_raw_vec()).unwrap();
//    image = image::imageops::resize(&image, h/ 4, w / 4, image::imageops::FilterType::Nearest);
//    println!("{:?}",image.width());
//    println!("{:?}",image.height());
//    let pixel_1x_arr = image_to_arr(image, (h / 4) as usize, (w/4) as usize);
//    pixel_1x_arr
//}

pub fn arr_to_image(arr: Array<f32, Ix4>, h:u32, w:u32) -> RgbImage{
    let mut arr_f : Array<f32, Ix3> = arr.slice(s![0,..,..,..]).to_owned();
    arr_f.swap_axes(1,2); //cxy->cyx
    arr_f.swap_axes(0,2); //cyx->xyc
    arr_f = arr_f.as_standard_layout().to_owned();
    arr_f = arr_f * 255.99_f32;
    let arr_i : Array<u8, Ix3> = arr_f.mapv(|elem| elem as u8);
    let image = RgbImage::from_raw(h, w, arr_i.into_raw_vec()).unwrap();
    //image::imageops::resize(&image, 250, 250, image::imageops::FilterType::Nearest);
    //image::imageops::resize(&image, 1000, 1000, image::imageops::FilterType::Nearest);
    image
}

pub enum Media{
    Frame((image::DynamicImage,(u8, u8, u8))),
    Frames((Vec<(image::DynamicImage, image::Delay)>,(u16, u16, u16))),
}

pub fn load_file(filename: String) -> Media{
    let frame_to_data = |x:image::Frame| {
        let delay = x.delay();
        let image = image::DynamicImage::ImageRgba8(x.into_buffer());
        (image, delay)
    };
    let frames_to_vec = |x : image::Frames| x.collect_frames().expect("Error decoding animation").into_iter().map(frame_to_data).collect();
    let format = image::ImageFormat::from_path(filename.clone()).expect("Unrecognized file type");
    use image::ImageFormat::{Png as Png, WebP as WebP, Gif as Gif};
            let file_in = File::open(filename.clone()).expect("File open failed");
    let media = match format {
        Png => {
            let info = pngchat::Png::from_file(filename.clone()).expect("File open failed");
            //println!("png info : {:?}",info.chunks());
            let bkgd = info.chunk_by_type(&"bKGD");
            //samples : https://www.nayuki.io/page/png-file-chunk-inspector
            //spec : http://www.libpng.org/pub/png/spec/1.2/PNG-Chunks.html
            let bgcolor = match bkgd {
                Some(x) => {
                    let chunk_data = x.data();
                    //println!("bKGD : {:?}", chunk_data);
                    //println!("{:?}", ((chunk_data[0] as u16) << 8) | chunk_data[1] as u16,); 
                    match chunk_data.len(){
                        1 => {
                            println!("Palette index in Background color (bKGD color type 3) is not supported yet. Using white color as background.");
                            (255,255,255)
                        },
                        2 => (
                            ((chunk_data[0] as u16) << 8) | chunk_data[1] as u16,
                            ((chunk_data[0] as u16) << 8) | chunk_data[1] as u16,
                            ((chunk_data[0] as u16) << 8) | chunk_data[1] as u16,
                            ),
                        3 => (
                            ((chunk_data[0] as u16) << 8) | chunk_data[1] as u16,
                            ((chunk_data[1] as u16) << 8) | chunk_data[2] as u16,
                            ((chunk_data[2] as u16) << 8) | chunk_data[3] as u16,
                            ),
                        _ => {
                            (255,255,255)
                        }
                    }
                },
                None => {
                    println!("No bKGD chunk was detected in PNG file. Using white as background color.");
                    (255,255,255)
                },
            };
            let decoder = image::codecs::png::PngDecoder::new(file_in).unwrap();
            let apng = decoder.is_apng();

            match decoder.is_apng(){
                true => {
                    //let frames = frames_to_vec(decoder.apng().into_frames());
                    let frames: Vec<(image::DynamicImage, image::Delay)> = frames_to_vec(decoder.apng().into_frames());
                    println!("Frames in image {:}", frames.len());
                    Media::Frames((frames, bgcolor))
                }
                false =>  {
                    let image = ImageReader::open(filename).expect("File open failed");
                    let x = image.decode().expect("File decode failed");
                    Media::Frame((x, (255,255,255)))
                }
            }
        }
        WebP => {
            let decoder = image::codecs::webp::WebPDecoder::new(file_in).unwrap();
            let frames = frames_to_vec(decoder.into_frames());
            Media::Frames((frames, (255,255,255)))
        }
        Gif => {
            let decoder = image::codecs::gif::GifDecoder::new(file_in).unwrap();
            let frames = frames_to_vec(decoder.into_frames());
            Media::Frames((frames, (255,255,255)))
        }
        _ => {
			let image = ImageReader::open(filename).expect("File open failed");
			let x = image.decode().expect("File decode failed");
			Media::Frame((x, (255,255,255)))
		}
	};
	media
}

