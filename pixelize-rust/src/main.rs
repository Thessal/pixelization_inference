fn main() {
    let _alias_RGBEnc = include_bytes!("../../alias_RGBEnc.onnx");
    let _alias_RGBDec = include_bytes!("../../alias_RGBDec.onnx");
    let _g_a_RGBEnc = include_bytes!("../../g_a_RGBEnc.onnx");
    let _g_a_PBEnc = include_bytes!("../../g_a_PBEnc.onnx");
    let _g_a_MLP = include_bytes!("../../g_a_MLP.onnx");
    let _g_a_RGBDec_mod_conv_1 = include_bytes!("../../g_a_RGBDec_mod_conv_1.onnx");
    let _g_a_RGBDec_mod_conv_2 = include_bytes!("../../g_a_RGBDec_mod_conv_2.onnx");
    let _g_a_RGBDec_mod_conv_3 = include_bytes!("../../g_a_RGBDec_mod_conv_3.onnx");
    let _g_a_RGBDec_mod_conv_4 = include_bytes!("../../g_a_RGBDec_mod_conv_4.onnx");
    let _g_a_RGBDec_mod_conv_5 = include_bytes!("../../g_a_RGBDec_mod_conv_5.onnx");
    let _g_a_RGBDec_mod_conv_6 = include_bytes!("../../g_a_RGBDec_mod_conv_6.onnx");
    let _g_a_RGBDec_mod_conv_7 = include_bytes!("../../g_a_RGBDec_mod_conv_7.onnx");
    let _g_a_RGBDec_mod_conv_8 = include_bytes!("../../g_a_RGBDec_mod_conv_8.onnx");
    let _g_a_RGBDec_upsample_block1 = include_bytes!("../../g_a_RGBDec_upsample_block1.onnx");
    let _g_a_RGBDec_upsample_block2 = include_bytes!("../../g_a_RGBDec_upsample_block2.onnx");
    let _g_a_RGBDec_conv_1 = include_bytes!("../../g_a_RGBDec_conv_1.onnx");
    let _g_a_RGBDec_conv_2 = include_bytes!("../../g_a_RGBDec_conv_2.onnx");
    let _g_a_RGBDec_conv_3 = include_bytes!("../../g_a_RGBDec_conv_3.onnx");
    let _reference = include_bytes!("../../reference.png");
    println!("Hello, world!");
}
