use std::{env, error::Error, io::Cursor};

use image::{io::Reader as ImageReader, Rgb32FImage};

use std::collections::HashMap;
use onnxruntime::{
    environment::Environment, 
    ndarray::{Array, Dim, Ix0, Ix1, Ix2, Ix3, Ix4, Ix5},
    tensor::OrtOwnedTensor, GraphOptimizationLevel,
    LoggingLevel, OrtError
};

fn load_model<'a>(environment:&'a Environment, bytes:&'a[u8]) -> Result<onnxruntime::session::Session<'a>, OrtError> {
    let mut session = environment
        .new_session_builder().unwrap()
        .with_optimization_level(GraphOptimizationLevel::Basic).unwrap()
        .with_number_threads(1).unwrap()
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

fn process_image(models: &mut HashMap<String, onnxruntime::session::Session<'_>>, data: Array<f32, Ix4>){
    //println!("{:?}",data);
    // TODO : preprocess
    let _model = models.get_mut(&("g_a_RGBEnc".to_string()));
    let model = _model.unwrap();
    let i = vec![data];
    let outputs : Vec<OrtOwnedTensor<f32, _>> = model.run(i).unwrap();
    println!("{:?}", outputs);
}

fn process_image_all(models: &mut HashMap<String, onnxruntime::session::Session<'_>>, image_path: Vec<String>) -> Result<(),()>{
    let reference_b = Cursor::new(include_bytes!("../../reference.png"));
    let reference_data = ImageReader::new(reference_b).with_guessed_format().unwrap().decode().unwrap();
    for filename in image_path{
        //let image = ImageReader::open("../input_file.jpg").expect("File open failed")
        let image = ImageReader::open("../red.jpg").expect("File open failed")
            .decode().expect("File decode failed").into_rgb32f();
        let mut data = Rgb32FImage::new(1000, 1000);
        image::imageops::tile(&mut data, &image);
        //println!("{:?}", data.dimensions());
        let mut data_arr = Array::from_iter(data.iter().map(|x| *x)).into_shape((1, 1000, 1000, 3)).unwrap();
        data_arr.swap_axes(1, 3);
        data_arr.swap_axes(2, 3);
        process_image(models, data_arr);
        println!("Not implemented");
    }
    Ok(())
}

fn main(){
    let args: Vec<String> = env::args().collect();
    if args.len() == 1{
        println!("TODO: Help message");
        ()
    }

    let environment = Environment::builder()
        .with_name("test")
        .with_log_level(LoggingLevel::Info)
        .build().unwrap();
    let mut models = load_model_all(&environment);

    process_image_all(&mut models, args[1..].to_vec());
}
