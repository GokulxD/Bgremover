from pyrogram import Client, filters
import torch
import torchvision.transforms as transforms
from PIL import Image
from io import BytesIO

app = Client("")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('facebookresearch/ESRGAN', 'esrgan', pretrained=True).to(device).eval()

def upscale_image(input_image):
    image = Image.open(BytesIO(input_image))
    image = transforms.ToTensor()(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output_image = model(image).data.squeeze().cpu().clamp(0, 1)
    
    output_image = transforms.ToPILImage()(output_image)
    
    output_buffer = BytesIO()
    output_image.save(output_buffer, format="JPG")
    
    return output_buffer.getvalue()
@app.on_message(filters.command("upscale"))
async def upscale_command(client, message):
    
    if message.photo:
        photo = message.photo[-1]
      
        input_image = await photo.download()
      
        upscaled_image = upscale_image(open(input_image, "rb").read())
        
        await message.reply_photo(upscaled_image)
app.run()
