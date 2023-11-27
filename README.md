# Fine-tuning Stable Diffusion XL on AWS for Generative AI-powered Product Concept Design

Source code files for the blog post: [Fine-tuning Stable Diffusion XL on AWS for Generative AI-powered Product Concept Design](https://garystafford.medium.com/fine-tuning-stable-diffusion-xl-on-aws-for-generative-ai-powered-product-concept-design-dae6f4c8c8fa). For more information, see the blog post. The SageMaker notebook, [SDXL_AutoTrain_DreamBooth.ipynb](SDXL_AutoTrain_DreamBooth.ipynb) contains all the code demonstrated in the post.

## LoRA Weights

To use the PyTorch LoRA weights with the SDXL 1.0 model, unzip the `mb_amg_gt_oue_dreambooth.zip` file. The resulting `mb_amg_gt_oue_dreambooth` and inclosed `pytorch_lora_weights.safetensors` file can be used with the SDXL 1.0 base model.

## Sample Concept Images

### Test of the Fine-tuned Model to Generate `oue car` Images

The photorealistic images below were generated using the following prompts:

```python
subject_prompt = subject_prompt = """oue, photo of oue car, sporty, fast, sleek, sexy, aggressive, high performance, daytime, futuristic cityscape"""

subject_negative_prompt = """person, people, human, rider, floating objects, text, words, writing, letters, phrases, trademark, watermark, icon, logo, banner, signature, username, monochrome, cropped, cut-off"""

refiner_prompt = """ultra-high-definition, photorealistic, 8k uhd, high-quality, ultra sharp detail"""

refiner_negative_prompt = """low quality, low-resolution, out of focus, blurry, grainy, artifacts, defects, jpeg artifacts, noise"""
```

<table border="0" cellspacing="10" cellpadding="10">
    <tr>
        <td>
            <img src="./image_samples/image_400.png" alt="DreamBooth" width="512"/>
        </td>
        <td>
            <img src="./image_samples/image_401.png" alt="DreamBooth" width="512"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="./image_samples/image_406.png" alt="DreamBooth" width="512"/>
        </td>
        <td>
            <img src="./image_samples/image_407.png" alt="DreamBooth" width="512"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="./image_samples/image_409.png" alt="DreamBooth" width="512"/>
        </td>
        <td>
            <img src="./image_samples/image_410.png" alt="DreamBooth" width="512"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="./image_samples/image_413.png" alt="DreamBooth" width="512"/>
        </td>
        <td>
            <img src="./image_samples/image_418.png" alt="DreamBooth" width="512"/>
        </td>
    </tr>
</table>

### Rough Product Sketches of `oue electric scooter`

The rough product sketches below were generated using the following prompts:

```python
subject_prompt = """oue, marker rendering of oue electric scooter, concept art, futuristic cityscape, high contrast, black and white, black marker, marker drawing, sketch, monochromatic illustration, illustrative, graphic, muted, expressive strokes"""

subject_negative_prompt = """person, people, human, rider, floating objects, colors, text, words, writing, letters, phrases, trademark, watermark, icon, logo, banner, signature, username, cropped, cut-off, patterned background"""

refiner_prompt = """sharp, crisp, in-focus, uncropped, high-quality"""

refiner_negative_prompt = """photographic, photo, photorealistic, low quality, low-resolution, out of focus, blurry, grainy, artifacts, defects, jpeg artifacts, noise"""
```

<table border="0" cellspacing="10" cellpadding="10">
    <tr>
        <td>
            <img src="./image_samples/image_100.png" alt="DreamBooth" width="512"/>
        </td>
        <td>
            <img src="./image_samples/image_108.png" alt="DreamBooth" width="512"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="./image_samples/image_110.png" alt="DreamBooth" width="512"/>
        </td>
        <td>
            <img src="./image_samples/image_111.png" alt="DreamBooth" width="512"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="./image_samples/image_112.png" alt="DreamBooth" width="512"/>
        </td>
        <td>
            <img src="./image_samples/image_113.png" alt="DreamBooth" width="512"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="./image_samples/image_115.png" alt="DreamBooth" width="512"/>
        </td>
        <td>
            <img src="./image_samples/image_119.png" alt="DreamBooth" width="512"/>
        </td>
    </tr>
</table>

Additional variations of rough product sketches generated with the latest LoRA weights and these prompts:

```python
subject_prompt = """oue, marker rendering of oue electric scooter, concept art, futuristic cityscape, high contrast, black and white, black marker, marker drawing, sketch, monochromatic illustration, illustrative, graphic, muted, expressive strokes"""

subject_negative_prompt = """person, people, human, rider, floating objects, colors, text, words, writing, letters, phrases, trademark, watermark, icon, logo, banner, signature, username, cropped, cut-off, patterned background"""

refiner_prompt = """sharp, crisp, in-focus, uncropped, high-quality"""

refiner_negative_prompt = """photographic, photo, photorealistic, low quality, low-resolution, out of focus, blurry, grainy, artifacts, defects, jpeg artifacts, noise"""
```

<table border="0" cellspacing="10" cellpadding="10">
    <tr>
        <td>
            <img src="./image_samples/image_600.png" alt="DreamBooth" width="512"/>
        </td>
        <td>
            <img src="./image_samples/image_601.png" alt="DreamBooth" width="512"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="./image_samples/image_602.png" alt="DreamBooth" width="512"/>
        </td>
        <td>
            <img src="./image_samples/image_603.png" alt="DreamBooth" width="512"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="./image_samples/image_604.png" alt="DreamBooth" width="512"/>
        </td>
        <td>
            <img src="./image_samples/image_605.png" alt="DreamBooth" width="512"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="./image_samples/image_606.png" alt="DreamBooth" width="512"/>
        </td>
        <td>
            <img src="./image_samples/image_607.png" alt="DreamBooth" width="512"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="./image_samples/image_608.png" alt="DreamBooth" width="512"/>
        </td>
        <td>
            <img src="./image_samples/image_609.png" alt="DreamBooth" width="512"/>
        </td>
    </tr>
</table>

### Color Marker Renderings of `oue electric scooter`

The color marker renderings below were generated using the following prompts:

```python
subject_prompt = """oue, color marker rendering of oue electric scooter, concept art, sporty, fast, sleek, sexy, aggressive, high performance, colors, urban, futuristic cityscape, marker, sketch, black and white lines, illustration, illustrative, marker drawing, expressive strokes, graphic"""

subject_negative_prompt = """person, people, human, rider, floating objects, text, words, writing, letters, phrases, trademark, watermark, icon, logo, banner, signature, username, monochrome, cropped, cut-off, patterned background, doubles, repeat elements"""

refiner_prompt = """sharp, crisp, in-focus, uncropped, high-quality"""

refiner_negative_prompt = """photographic, photo, photorealistic, low quality, low-resolution, out of focus, blurry, grainy, artifacts, defects, jpeg artifacts, noise"""
```

<table border="0" cellspacing="10" cellpadding="10">
    <tr>
        <td>
            <img src="./image_samples/image_001.png" alt="DreamBooth" width="512"/>
        </td>
        <td>
            <img src="./image_samples/image_002.png" alt="DreamBooth" width="512"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="./image_samples/image_003.png" alt="DreamBooth" width="512"/>
        </td>
        <td>
            <img src="./image_samples/image_004.png" alt="DreamBooth" width="512"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="./image_samples/image_005.png" alt="DreamBooth" width="512"/>
        </td>
        <td>
            <img src="./image_samples/image_006.png" alt="DreamBooth" width="512"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="./image_samples/image_007.png" alt="DreamBooth" width="512"/>
        </td>
        <td>
            <img src="./image_samples/image_008.png" alt="DreamBooth" width="512"/>
        </td>
    </tr>
</table>

Additional variations of color marker renderings generated with the latest LoRA weights and these prompts:

```python
subject_prompt = """oue, marker rendering of oue electric scooter, concept art, futuristic cityscape, solid color background, bright vibrant colors, marker, sketch, illustration, illustrative, marker drawing, expressive strokes, graphic"""

subject_negative_prompt = """person, people, human, rider, floating objects, text, words, writing, letters, phrases, trademark, watermark, icon, logo, banner, signature, username, monochrome, cropped, cut-off, patterned background"""

refiner_prompt = """sharp, crisp, in-focus, uncropped, high-quality"""

refiner_negative_prompt = """photographic, photo, photorealistic, low quality, low-resolution, out of focus, blurry, grainy, artifacts, defects, jpeg artifacts, noise"""
```

<table border="0" cellspacing="10" cellpadding="10">
    <tr>
        <td>
            <img src="./image_samples/image_500.png" alt="DreamBooth" width="512"/>
        </td>
        <td>
            <img src="./image_samples/image_501.png" alt="DreamBooth" width="512"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="./image_samples/image_502.png" alt="DreamBooth" width="512"/>
        </td>
        <td>
            <img src="./image_samples/image_503.png" alt="DreamBooth" width="512"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="./image_samples/image_504.png" alt="DreamBooth" width="512"/>
        </td>
        <td>
            <img src="./image_samples/image_506.png" alt="DreamBooth" width="512"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="./image_samples/image_507.png" alt="DreamBooth" width="512"/>
        </td>
        <td>
            <img src="./image_samples/image_508.png" alt="DreamBooth" width="512"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="./image_samples/image_509.png" alt="DreamBooth" width="512"/>
        </td>
        <td>
            <img src="./image_samples/image_510.png" alt="DreamBooth" width="512"/>
        </td>
    </tr>
</table>

### Photorealistic Images of `oue electric scooter`

The photorealistic images below were generated using the following prompts:

```python
subject_prompt = """oue, photo of a oue electric scooter, sleek, smooth curves, colorful, daytime, urban, futuristic cityscape"""

subject_negative_prompt = """person, people, human, rider, floating objects, text, words, writing, letters, phrases, trademark, watermark, icon, logo, banner, signature, username, monochrome, cropped, cut-off, patterned background"""

refiner_prompt = """ultra-high-definition, photorealistic, 8k uhd, high-quality, ultra sharp detail"""

refiner_negative_prompt = """low quality, low-resolution, out of focus, blurry, grainy, artifacts, defects, jpeg artifacts, noise"""
```

<table border="0" cellspacing="10" cellpadding="10">
    <tr>
        <td>
            <img src="./image_samples/image_200.png" alt="DreamBooth" width="512"/>
        </td>
        <td>
            <img src="./image_samples/image_202.png" alt="DreamBooth" width="512"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="./image_samples/image_205.png" alt="DreamBooth" width="512"/>
        </td>
        <td>
            <img src="./image_samples/image_206.png" alt="DreamBooth" width="512"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="./image_samples/image_207.png" alt="DreamBooth" width="512"/>
        </td>
        <td>
            <img src="./image_samples/image_209.png" alt="DreamBooth" width="512"/>
        </td>
    </tr>
</table>
