#![allow(non_snake_case)]

use std::collections::HashMap;
use onnxruntime::{
    environment::Environment,
    GraphOptimizationLevel, OrtError
};


fn load_model<'a>(environment:&'a Environment, bytes:&'a[u8]) -> Result<onnxruntime::session::Session<'a>, OrtError> {
    let session = environment
        .new_session_builder().unwrap()
        .with_optimization_level(GraphOptimizationLevel::Basic).unwrap()
        .with_number_threads((num_cpus::get() as usize).try_into().unwrap()).unwrap()
        .with_model_from_memory(bytes);
    session
}

pub fn load_model_all<'a>(environment:&'a Environment)-> HashMap<std::string::String, onnxruntime::session::Session<'a>>{
    let mut model_data = HashMap::<String, &[u8]>::new();

    // Pixelizer
    model_data.insert("alias_RGBEnc".to_string(), include_bytes!("../../alias_RGBEnc.onnx"));
    model_data.insert("alias_RGBDec".to_string(), include_bytes!("../../alias_RGBDec.onnx"));
    model_data.insert("g_a_RGBEnc".to_string(), include_bytes!("../../g_a_RGBEnc.onnx"));
    model_data.insert("g_a_PBEnc".to_string(), include_bytes!("../../g_a_PBEnc.onnx"));
    model_data.insert("g_a_MLP".to_string(), include_bytes!("../../g_a_MLP.onnx"));
    model_data.insert("g_a_RGBDec_mod_conv_1".to_string(), include_bytes!("../../g_a_RGBDec_mod_conv_1.onnx"));
    model_data.insert("g_a_RGBDec_mod_conv_2".to_string(), include_bytes!("../../g_a_RGBDec_mod_conv_2.onnx"));
    model_data.insert("g_a_RGBDec_upsample_block1".to_string(), include_bytes!("../../g_a_RGBDec_upsample_block1.onnx"));
    model_data.insert("g_a_RGBDec_upsample_block2".to_string(), include_bytes!("../../g_a_RGBDec_upsample_block2.onnx"));
    model_data.insert("g_a_RGBDec_conv_1".to_string(), include_bytes!("../../g_a_RGBDec_conv_1.onnx"));
    model_data.insert("g_a_RGBDec_conv_2".to_string(), include_bytes!("../../g_a_RGBDec_conv_2.onnx"));
    model_data.insert("g_a_RGBDec_conv_3".to_string(), include_bytes!("../../g_a_RGBDec_conv_3.onnx"));

    // FastDVDNet
    model_data.insert("block1".to_string(), include_bytes!("../../block1.onnx"));
    model_data.insert("block2".to_string(), include_bytes!("../../block2.onnx"));
    model_data.insert("block1_256".to_string(), include_bytes!("../../block1_256.onnx"));
    model_data.insert("block2_256".to_string(), include_bytes!("../../block2_256.onnx"));

    let model_names = model_data.keys().cloned();
    let models = model_names
        .map(
            |i| (i.clone(), load_model(&environment, model_data[&i]).expect("Model Load Error"))
            ).collect::<HashMap<String, onnxruntime::session::Session<'a>>>();
    models
}
