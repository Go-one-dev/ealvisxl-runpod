import runpod
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "SG161222/RealVisXL_V3.0_Turbo",
    torch_dtype=torch.float16,
    use_safetensors=True
).to("cuda")

def run_inference(job):
    input_data = job["input"]
    prompt = input_data.get("prompt", "a fantasy castle")
    negative_prompt = input_data.get("negative_prompt", None)
    width = input_data.get("width", 1024)
    height = input_data.get("height", 1024)
    guidance_scale = input_data.get("guidance_scale", 7)

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_inference_steps=25
    ).images[0]

    output_path = "/tmp/output.png"
    image.save(output_path)

    return {"image_path": output_path}

runpod.serverless.start({"handler": run_inference})
