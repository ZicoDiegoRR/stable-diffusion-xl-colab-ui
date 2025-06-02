# A modified version of inference_realesrgan.py from https://github.com/xinntao/Real-ESRGAN/blob/master/inference_realesrgan.py for project purpose
# The libraries will be downloaded after running the first cell 
   
import cv2
import glob
import os
import ipywidgets as widgets
from basicsr.archs.rrdbnet_arch import RRDBNet
from diffusers.utils import load_image, make_image_grid
from basicsr.utils.download_util import load_file_from_url
from IPython.display import display, clear_output

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

class ESRGANWidget:
    # upload handler
    def input_upload_handler(self, change):
        if not os.path.exists("/content/upscale"):
            os.mkdir("/content/upscale")
        for filename, file_info in self.input_upload.value.items():
            with open(f"/content/upscale/{filename}", "wb") as up:
                up.write(file_info["content"])
            self.input_link.value = f"/content/upscale/{filename}"
                
    #creating widgets
    def __init__(self):
        self.warning_upscale = widgets.HTML(value="It's recommended to upscale any image with up to 1024x1024 resolution or lower to avoid high VRAM usage.")
        self.input_link = widgets.Text(placeholder="Image link or path")
        self.input_upload = widgets.FileUpload(accept="image/*", multiple=False)
        self.ersgan_input = widgets.VBox([self.warning_upscale, widgets.HBox([self.input_link, self.input_upload])])
        
        self.input_upload.observe(self.input_upload_handler, names="value")
        
        self.model_name = widgets.Dropdown(options=["RealESRGAN_x4plus", "RealESRNet_x4plus", "RealESRGAN_x4plus_anime_6B", "RealESRGAN_x2plus",
            "realesr-animevideov3", "realesr-general-x4v3"], description="Model")
        self.denoising = widgets.FloatSlider(min=0.1, max=1.0, step=0.01, description="Denoising Strength")
        self.upscale_factor = widgets.IntSlider(min=1, max=4, step=1, description="Upscale Factor")
        self.tile_size = widgets.IntText(description="Tile Size", value=0)
        self.tile_padding = widgets.IntText(description="Tile Padding", value=10)
        self.pre_padding = widgets.IntText(description="Pre-padding Size", value=0)
        self.face = widgets.Checkbox(description="Face Enhance", value=False)
        self.upsampler = widgets.Dropdown(options=["realesrgan", "bicubic"], description="Alpha Upsampler")
        self.ersgan_settings = widgets.VBox([
            self.ersgan_input,
            self.model_name,
            widgets.HBox([self.denoising, self.upscale_factor]),
            widgets.HBox([self.tile_size, self.tile_padding, self.pre_padding]),
            self.face,
            self.upsampler
        ])
        
    def execute_realesrgan(self, tab):
        # execute
        run_upscaling(
            tab=tab,
            input=self.input_link.value,
            model_name=self.model_name.value,
            denoise_strength=self.denoising.value,
            outscale=self.upscale_factor.value,
            tile=self.tile_size.value,
            tile_pad=self.tile_padding.value,
            pre_pad=self.pre_padding.value,
            face_enhance=self.face.value,
            alpha_upsampler=self.upsampler.value
        )

#too lazy to edit the run_upscaling() function
class VariableHandlerESRGAN:
    
    def variable_constructor(self, 
        input, 
        model_name, 
        output, 
        denoise_strength,
        outscale, 
        model_path,
        tile, 
        tile_pad, 
        suffix="",
        pre_pad=0,
        face_enhance=False, 
        fp32=False,
        alpha_upsampler="realesrgan",
        ext="auto",
        gpu_id=None):
             
        self.input = input
        self.model_name = model_name
        self.output = output
        self.denoise_strength = denoise_strength
        self.outscale = outscale
        self.model_path = model_path
        self.suffix = ""
        self.tile = tile
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        self.face_enhance = face_enhance 
        self.fp32 = fp32
        self.alpha_upsampler = alpha_upsampler
        self.ext = ext
        self.gpu_id = gpu_id
        
def run_upscaling(
    tab,
    input, 
    model_name, 
    tile,
    tile_pad,
    outscale,
    denoise_strength,
    output="/content/Upscaled/" if not os.path.exists("/content/gdrive/MyDrive") else "/content/gdrive/MyDrive/Upscaled", 
    model_path=None, 
    suffix="",
    pre_pad=0,
    face_enhance=False, 
    fp32=False,
    alpha_upsampler="realesrgan",
    ext="auto",
    gpu_id=None):

    args = VariableHandlerESRGAN()
        
    args.variable_constructor(
        input=input, 
        model_name=model_name, 
        output=output,
        denoise_strength=denoise_strength,
        outscale=outscale, 
        model_path=model_path,
        tile=tile, 
        tile_pad=tile_pad, 
        pre_pad=pre_pad,
        face_enhance=face_enhance, 
        alpha_upsampler=alpha_upsampler
    )

    # determine models according to model names
    args.model_name = args.model_name.split('.')[0]
    netscale = args.outscale
    if args.model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        # netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
    elif args.model_name == 'RealESRNet_x4plus':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        # netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']
    elif args.model_name == 'RealESRGAN_x4plus_anime_6B':  # x4 RRDBNet model with 6 blocks
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        # netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
    elif args.model_name == 'RealESRGAN_x2plus':  # x2 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        # netscale = 2
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
    elif args.model_name == 'realesr-animevideov3':  # x4 VGG-style model (XS size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        # netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']
    elif args.model_name == 'realesr-general-x4v3':  # x4 VGG-style model (S size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        # netscale = 4
        file_url = [
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
        ]

    # determine model paths
    if args.model_path is not None:
        model_path = args.model_path
    else:
        model_path = os.path.join('/content/RealESRGAN/weights', args.model_name + '.pth')
        if not os.path.isfile(model_path):
            ROOT_DIR = "/content/RealESRGAN"
            for url in file_url:
                # model_path will be updated
                model_path = load_file_from_url(
                    url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)

    # use dni to control the denoise strength
    dni_weight = None
    if args.model_name == 'realesr-general-x4v3' and args.denoise_strength != 1:
        wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
        model_path = [model_path, wdn_model_path]
        dni_weight = [args.denoise_strength, 1 - args.denoise_strength]

    # restorer
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        half=not args.fp32,
        gpu_id=args.gpu_id)

    if args.face_enhance:  # Use GFPGAN for face enhancement
        from gfpgan import GFPGANer
        face_enhancer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
            upscale=args.outscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=upsampler)
    os.makedirs(args.output, exist_ok=True)

    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(glob.glob(os.path.join(args.input, '*')))

    for idx, path in enumerate(paths):
        imgname, extension = os.path.splitext(os.path.basename(path))

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 3 and img.shape[2] == 4:
            img_mode = 'RGBA'
        else:
            img_mode = None

        try:
            if args.face_enhance:
                _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
            else:
                output, _ = upsampler.enhance(img, outscale=args.outscale)
        except RuntimeError as error:
            print('Error', error)
        else:
            if args.ext == 'auto':
                extension = extension[1:]
            else:
                extension = args.ext
            if img_mode == 'RGBA':  # RGBA images should be saved in png format
                extension = 'png'
            img_filename_with_prompt = f"[Upscaled] {imgname}"
            if (len(img_filename_with_prompt) + len(extension) + 1) > 255:
                img_filename = f"{img_filename_with_prompt[:245]}.{extension}"
            else:
                img_filename = f"{img_filename_with_prompt}.{extension}"
               
            save_path = os.path.join(args.output, img_filename)
            cv2.imwrite(save_path, output)
            input_width, input_height = load_image(input).size
            output_width, output_height = load_image(save_path).size
            clear_output()
            display(tab)
            display(make_image_grid([load_image(input), load_image(save_path).resize((input_width, input_height))], rows=1, cols=2))
            print(f"Original resolution: {input_width}x{input_height} px")
            print(f"Upscaled resolution: {output_width}x{output_height} px")
            print(f"Image is saved at {save_path}.")
