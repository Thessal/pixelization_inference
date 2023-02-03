#![allow(non_snake_case)]
//#![allow(non_camel_case_types)]

//use std::{env, error::Error, io::Cursor, ops::Mul, ops::Add, ops::Sub, ops::Div};
use std::{env, io::Cursor, ops::Mul, ops::Sub, };

use image::{io::Reader as ImageReader, Rgb32FImage, RgbImage};
use image;

use std::collections::HashMap;
use onnxruntime::{
    environment::Environment, 
    //ndarray::{Array, Dim, Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, IxDyn, s, ArrayView},
    ndarray::{Array, Ix2, Ix3, Ix4, IxDyn, s, },
    //tensor::{OrtOwnedTensor, FromArray, InputTensor},
    //tensor::{OrtOwnedTensor},
    GraphOptimizationLevel, LoggingLevel, OrtError
};


fn load_model<'a>(environment:&'a Environment, bytes:&'a[u8]) -> Result<onnxruntime::session::Session<'a>, OrtError> {
    let session = environment
        .new_session_builder().unwrap()
        .with_optimization_level(GraphOptimizationLevel::Basic).unwrap()
        .with_number_threads((num_cpus::get() as usize).try_into().unwrap()).unwrap()
        .with_model_from_memory(bytes);
    session
}

fn load_model_all<'a>(environment:&'a Environment)-> HashMap<std::string::String, onnxruntime::session::Session<'a>>{
    let mut model_data = HashMap::<String, &[u8]>::new();
    model_data.insert("alias_RGBEnc".to_string(), include_bytes!("../../alias_RGBEnc.onnx"));
    model_data.insert("alias_RGBDec".to_string(), include_bytes!("../../alias_RGBDec.onnx"));
    model_data.insert("g_a_RGBEnc".to_string(), include_bytes!("../../g_a_RGBEnc.onnx"));
    model_data.insert("g_a_PBEnc".to_string(), include_bytes!("../../g_a_PBEnc.onnx"));
    model_data.insert("g_a_MLP".to_string(), include_bytes!("../../g_a_MLP.onnx"));
    model_data.insert("g_a_RGBDec_mod_conv_1".to_string(), include_bytes!("../../g_a_RGBDec_mod_conv_1.onnx"));
    model_data.insert("g_a_RGBDec_mod_conv_2".to_string(), include_bytes!("../../g_a_RGBDec_mod_conv_2.onnx"));
    model_data.insert("g_a_RGBDec_mod_conv_3".to_string(), include_bytes!("../../g_a_RGBDec_mod_conv_3.onnx"));
    model_data.insert("g_a_RGBDec_mod_conv_4".to_string(), include_bytes!("../../g_a_RGBDec_mod_conv_4.onnx"));
    model_data.insert("g_a_RGBDec_mod_conv_5".to_string(), include_bytes!("../../g_a_RGBDec_mod_conv_5.onnx"));
    model_data.insert("g_a_RGBDec_mod_conv_6".to_string(), include_bytes!("../../g_a_RGBDec_mod_conv_6.onnx"));
    model_data.insert("g_a_RGBDec_mod_conv_7".to_string(), include_bytes!("../../g_a_RGBDec_mod_conv_7.onnx"));
    model_data.insert("g_a_RGBDec_mod_conv_8".to_string(), include_bytes!("../../g_a_RGBDec_mod_conv_8.onnx"));
    model_data.insert("g_a_RGBDec_upsample_block1".to_string(), include_bytes!("../../g_a_RGBDec_upsample_block1.onnx"));
    model_data.insert("g_a_RGBDec_upsample_block2".to_string(), include_bytes!("../../g_a_RGBDec_upsample_block2.onnx"));
    model_data.insert("g_a_RGBDec_conv_1".to_string(), include_bytes!("../../g_a_RGBDec_conv_1.onnx"));
    model_data.insert("g_a_RGBDec_conv_2".to_string(), include_bytes!("../../g_a_RGBDec_conv_2.onnx"));
    model_data.insert("g_a_RGBDec_conv_3".to_string(), include_bytes!("../../g_a_RGBDec_conv_3.onnx"));

    let model_names = model_data.keys().cloned();
    let models = model_names
        .map(
            |i| (i.clone(), load_model(&environment, model_data[&i]).expect("Model Load Error"))
            ).collect::<HashMap<String, onnxruntime::session::Session<'a>>>();
    models
}

fn grayscale(data: Array<f32, Ix4>)->Array<f32, Ix4>{
    // ITU-R 601-2 : L = R * 299/1000 + G * 587/1000 + B * 114/1000
    let R = data.slice(s![..,0..1,..,..]);
    let G = data.slice(s![..,1..2,..,..]);
    let B = data.slice(s![..,2..3,..,..]);
    let L : Array<f32, Ix4> = R.mul(0.299_f32) + G.mul(0.587_f32) + B.mul(0.114_f32);
    let L_arr = ndarray::concatenate(ndarray::Axis(1), &[L.view(), L.view(), L.view()]).unwrap();
    //let mut L_arr = Array::<f32, Ix4>::zeros(data.raw_dim());
    //L_arr.slice_mut(s![..,0..1,..,..]).assign(&L);
    //L_arr.slice_mut(s![..,1..2,..,..]).assign(&L);
    //L_arr.slice_mut(s![..,2..3,..,..]).assign(&L);
    L_arr
}

fn normalize(data: Array<f32, Ix4>)->Array<f32, Ix4>{
    let R = data.slice(s![..,0..1,..,..]);
    let G = data.slice(s![..,1..2,..,..]);
    let B = data.slice(s![..,2..3,..,..]);
    //let R_norm : Array<f32, Ix4> = (R.sub(R.mean().unwrap()))/(R.std(0.0_f32).max(0.1_f32));
    //let G_norm : Array<f32, Ix4> = (G.sub(G.mean().unwrap()))/(G.std(0.0_f32).max(0.1_f32));
    //let B_norm : Array<f32, Ix4> = (B.sub(B.mean().unwrap()))/(B.std(0.0_f32).max(0.1_f32));
    //let R_norm : Array<f32, Ix4> = ((R.div(255.0_f32))-0.5_f32) * 1.4_f32;
    //let G_norm : Array<f32, Ix4> = ((G.div(255.0_f32))-0.5_f32) * 1.4_f32; 
    //let B_norm : Array<f32, Ix4> = ((B.div(255.0_f32))-0.5_f32) * 1.4_f32;
    let R_norm : Array<f32, Ix4> = R.sub(0.5_f32) * 1.4_f32;
    let G_norm : Array<f32, Ix4> = G.sub(0.5_f32) * 1.4_f32; 
    let B_norm : Array<f32, Ix4> = B.sub(0.5_f32) * 1.4_f32;
    let N_arr = ndarray::concatenate(ndarray::Axis(1), &[R_norm.view(), G_norm.view(), B_norm.view()]).unwrap();
    N_arr
}



fn g_a_RGBDec(models: &mut HashMap<String, onnxruntime::session::Session<'_>>, 
              x:Array<f32, Ix4>, code:Array<f32, Ix2>) -> Array<f32, Ix4>{
    let mut x_ : Array<f32, IxDyn> = x.clone().into_dyn();
    let code_0 : Array<f32, IxDyn> = code.slice(s![..,..256]).to_owned().into_dyn();
    let code_1 : Array<f32, IxDyn> = code.slice(s![..,256*1..256*2]).to_owned().into_dyn();
    let code_2 : Array<f32, IxDyn> = code.slice(s![..,256*2..256*3]).to_owned().into_dyn();
    let code_3 : Array<f32, IxDyn> = code.slice(s![..,256*3..256*4]).to_owned().into_dyn();
    let code_4 : Array<f32, IxDyn> = code.slice(s![..,256*4..256*5]).to_owned().into_dyn();
    let code_5 : Array<f32, IxDyn> = code.slice(s![..,256*5..256*6]).to_owned().into_dyn();
    let code_6 : Array<f32, IxDyn> = code.slice(s![..,256*6..256*7]).to_owned().into_dyn();
    let code_7 : Array<f32, IxDyn> = code.slice(s![..,256*7..256*8]).to_owned().into_dyn();
    {
        let residual = x_.clone();
        {
            let g_a_RGBDec_mod_conv_1 = models.get_mut(&("g_a_RGBDec_mod_conv_1".to_string())).unwrap();
            x_ = (*g_a_RGBDec_mod_conv_1.run::<'_, '_, '_, f32, f32, IxDyn>( vec![x_, code_0 ]).unwrap()[0]).to_owned();
        }
        {
            let g_a_RGBDec_mod_conv_2 = models.get_mut(&("g_a_RGBDec_mod_conv_2".to_string())).unwrap();
            x_ = (*g_a_RGBDec_mod_conv_2.run::<'_, '_, '_, f32, f32, IxDyn>( vec![x_, code_1 ]).unwrap()[0]).to_owned();
        }
        x_ = x_ + residual;
    }
    {
        let residual = x_.clone();
        {
            let g_a_RGBDec_mod_conv_2 = models.get_mut(&("g_a_RGBDec_mod_conv_2".to_string())).unwrap();
            x_ = (*g_a_RGBDec_mod_conv_2.run::<'_, '_, '_, f32, f32, IxDyn>( vec![x_, code_2 ]).unwrap()[0]).to_owned();
            x_ = (*g_a_RGBDec_mod_conv_2.run::<'_, '_, '_, f32, f32, IxDyn>( vec![x_, code_3 ]).unwrap()[0]).to_owned();
        }
        x_ = x_ + residual;
    }
    {
        let residual = x_.clone();
        {
            let g_a_RGBDec_mod_conv_2 = models.get_mut(&("g_a_RGBDec_mod_conv_2".to_string())).unwrap();
            x_ = (*g_a_RGBDec_mod_conv_2.run::<'_, '_, '_, f32, f32, IxDyn>( vec![x_, code_4 ]).unwrap()[0]).to_owned();
            x_ = (*g_a_RGBDec_mod_conv_2.run::<'_, '_, '_, f32, f32, IxDyn>( vec![x_, code_5 ]).unwrap()[0]).to_owned();
        }
        x_ = x_ + residual;
    }
    {
        let residual = x_.clone();
        {
            let g_a_RGBDec_mod_conv_2 = models.get_mut(&("g_a_RGBDec_mod_conv_2".to_string())).unwrap();
            x_ = (*g_a_RGBDec_mod_conv_2.run::<'_, '_, '_, f32, f32, IxDyn>( vec![x_, code_6 ]).unwrap()[0]).to_owned();
            x_ = (*g_a_RGBDec_mod_conv_2.run::<'_, '_, '_, f32, f32, IxDyn>( vec![x_, code_7 ]).unwrap()[0]).to_owned();
        }
        x_ = x_ + residual;
    }
    {
        let g_a_RGBDec_upsample_block1 = models.get_mut(&("g_a_RGBDec_upsample_block1".to_string())).unwrap();
        x_ = (*g_a_RGBDec_upsample_block1.run::<'_, '_, '_, f32, f32, IxDyn>( vec![x_]).unwrap()[0]).to_owned();
    }
    {
        let g_a_RGBDec_conv_1 = models.get_mut(&("g_a_RGBDec_conv_1".to_string())).unwrap();
        x_ = (*g_a_RGBDec_conv_1.run::<'_, '_, '_, f32, f32, IxDyn>( vec![x_]).unwrap()[0]).to_owned();
    }
    {
        let g_a_RGBDec_upsample_block2 = models.get_mut(&("g_a_RGBDec_upsample_block2".to_string())).unwrap();
        x_ = (*g_a_RGBDec_upsample_block2.run::<'_, '_, '_, f32, f32, IxDyn>( vec![x_]).unwrap()[0]).to_owned();
    }
    {
        let g_a_RGBDec_conv_2 = models.get_mut(&("g_a_RGBDec_conv_2".to_string())).unwrap();
        x_ = (*g_a_RGBDec_conv_2.run::<'_, '_, '_, f32, f32, IxDyn>( vec![x_]).unwrap()[0]).to_owned();
    }
    {
        let g_a_RGBDec_conv_3 = models.get_mut(&("g_a_RGBDec_conv_3".to_string())).unwrap();
        x_ = (*g_a_RGBDec_conv_3.run::<'_, '_, '_, f32, f32, IxDyn>( vec![x_]).unwrap()[0]).to_owned();
    }

    x_.into_dimensionality::<Ix4>().unwrap()
}

fn g_a<'a>(models: &mut HashMap<String, onnxruntime::session::Session<'_>>, 
       clipart:Array<f32, Ix4>, pixelart:Array<f32, Ix4>) -> Array<f32, Ix4>{
    let clipart_ = vec![clipart];
    let pixelart_ = vec![pixelart];
    let feature : Array<f32, Ix4>;
    let code : Array<f32, IxDyn>;
    let adain_params : Array<f32, Ix2>;
    {
        let g_a_RGBEnc = models.get_mut(&("g_a_RGBEnc".to_string())).unwrap();
        feature = (*g_a_RGBEnc.run( clipart_ ).unwrap()[0]).to_owned().into_dimensionality::<Ix4>().unwrap();
    }
    {
        let g_a_PBEnc = models.get_mut(&("g_a_PBEnc".to_string())).unwrap();
        code = (*g_a_PBEnc.run(pixelart_).unwrap()[0]).to_owned();
    }
    {
        let g_a_MLP = models.get_mut(&("g_a_MLP".to_string())).unwrap();
        adain_params = (*g_a_MLP.run(vec![code]).unwrap()[0]).to_owned().into_dimensionality::<Ix2>().unwrap();
    }
    let images : Array<f32, Ix4> = g_a_RGBDec(models, feature, adain_params).into_dimensionality::<Ix4>().unwrap();
    images
}

fn process_image(models: &mut HashMap<String, onnxruntime::session::Session<'_>>, 
                 data: Array<f32, Ix4>, reference: Array<f32, Ix4>) -> Array<f32, Ix4> {
    let mut images : Array<f32, IxDyn> = g_a(models, data, reference).into_dyn();
    {
        let alias_RGBEnc = models.get_mut(&("alias_RGBEnc".to_string())).unwrap();
        images = (*alias_RGBEnc.run( vec![images] ).unwrap()[0]).to_owned()
    }
    {
        let alias_RGBDec = models.get_mut(&("alias_RGBDec".to_string())).unwrap();
        images = (*alias_RGBDec.run( vec![images] ).unwrap()[0]).to_owned()
    }
    let output : Array<f32, Ix4> = images.into_dimensionality::<Ix4>().unwrap();
    output 
}

fn image_to_arr(image: Rgb32FImage, h:usize, w:usize) -> Array<f32,Ix4> {
    let mut arr = Array::from_iter(image.iter().map(|x| *x)).into_shape((1, h, w, 3)).unwrap();
    arr.swap_axes(1, 3); //hwc->cwh
    arr.swap_axes(2, 3); //cwh->chw
    arr = arr.as_standard_layout().to_owned();
    //arr *= 255.0_f32;
    arr
}

fn arr_to_image(arr: Array<f32, Ix4>, h:u32, w:u32) -> RgbImage{
    let mut arr_f : Array<f32, Ix3> = arr.slice(s![0,..,..,..]).to_owned();
    arr_f.swap_axes(1,2); //cxy->cyx
    arr_f.swap_axes(0,2); //cyx->xyc
    arr_f = arr_f.as_standard_layout().to_owned();
    arr_f = (arr_f + 1.0_f32) * 0.5_f32 * 255.0_f32;
    let arr_i : Array<u8, Ix3> = arr_f.mapv(|elem| elem as u8);
    let image = RgbImage::from_raw(h, w, arr_i.into_raw_vec()).unwrap();
    image::imageops::resize(&image, 250, 250, image::imageops::FilterType::Nearest);
    image::imageops::resize(&image, 1000, 1000, image::imageops::FilterType::Nearest);
    image
}

enum Frames<'a>{
    Image(image::DynamicImage),
    Frames(image::Frames<'a>),
}

fn load_file(filename: String) -> image::DynamicImage{
    let format = image::ImageFormat::from_path(filename.clone()).expect("Unrecognized file type");
	let image_ =
    match format {
        image::ImageFormat::Png => {
			println!("png");
			ImageReader::open(filename.clone()).expect("File open failed").decode().expect("File decode failed")
		}
		image::ImageFormat::Gif | image::ImageFormat::WebP => {
			println!("Gif or WebP");
			ImageReader::open(filename.clone()).expect("File open failed").decode().expect("File decode failed")
        }
		image::ImageFormat::Jpeg | image::ImageFormat::WebP => {
			println!("JPG");
			ImageReader::open(filename.clone()).expect("File open failed").decode().expect("File decode failed")
        }
        _ => ImageReader::open(filename.clone()).expect("File open failed").decode().expect("File decode failed")
    };
        //get_frames
    let image = ImageReader::open(filename.clone()).expect("File open failed")
                    .decode().expect("File decode failed");
    image
}

fn process_image_all(models: &mut HashMap<String, onnxruntime::session::Session<'_>>, image_path: Vec<String>) -> Result<(),()>{
    let reference_b = Cursor::new(include_bytes!("../../reference.png"));
    let reference_image = ImageReader::new(reference_b).with_guessed_format().unwrap().decode().unwrap().into_rgb32f();
    let reference_arr = normalize(grayscale(image_to_arr(reference_image, 256, 256)));
    for filename in image_path{
        println!("{}", filename);
        //let image = ImageReader::open(filename.clone()).expect("File open failed")
        //    .decode().expect("File decode failed").into_rgb32f();
        let image = load_file(filename.clone()).into_rgb32f();
        //let mut data = Rgb32FImage::new(1000, 1000);
        let mut data = Rgb32FImage::from_pixel(1000, 1000, image::Rgb::<f32>([0.5_f32,0.5_f32,0.5_f32]));
        //let mut data = Rgb32FImage::from_pixel(1000, 1000, [0.5,0.5,0.5]);
        image::imageops::overlay(&mut data, &image, 0, 0);
        let data_arr = normalize(image_to_arr(data, 1000, 1000));
        let output: Array<f32, Ix4> = process_image(models, data_arr, reference_arr.clone());
        let mut img_p = arr_to_image(output, 1000, 1000);
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
    let mut models = load_model_all(&environment);

    process_image_all(&mut models, args[1..].to_vec()).expect("Main loop failed");
}
