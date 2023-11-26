# Fine-tuning Stable Diffusion XL on AWS for Generative AI-powered Product Concept Design

Source code files for the blog post: [Fine-tuning Stable Diffusion XL on AWS for Generative AI-powered Product Concept Design](https://garystafford.medium.com/fine-tuning-stable-diffusion-xl-on-aws-for-generative-ai-powered-product-concept-design-dae6f4c8c8fa). For more information, see the blog post. The SageMaker notebook, [SDXL_AutoTrain_DreamBooth.ipynb](SDXL_AutoTrain_DreamBooth.ipynb) contains all the code demonstrated in the post.

## LoRA Weights

To use the PyTorch LoRA weights with the SDXL 1.0 model, unzip the `./amg_gt_oue_dreambooth/pytorch_lora_weights.safetensors.zip` file. The resulting `pytorch_lora_weights.safetensors` file should be placed in the `sdxl_dreambooth_autotrain_demo/amg_gt_oue_dreambooth` directory.

## Sample Concept Images

<table>
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
</table>
