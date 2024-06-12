import diffusers
import torch
import random
from tqdm import tqdm
from constants import SUBJECTS, MEDIUMS
from PIL import Image

class CLIPSlider:
    def __init__(
            self,
            sd_pipe,
            device: torch.device,
            target_word: str,
            opposite: str
    ):

        self.device = device
        self.pipe = sd_pipe.to(self.device)
        self.avg_diff = self.find_latent_direction(target_word, opposite)

    def find_latent_direction(self,
                              target_word:str,
                              opposite:str):

        # lets identify a latent direction by taking differences between opposites
        # target_word = "happy"
        # opposite = "sad"

        num = 300

        with torch.no_grad():
            positives = []
            negatives = []
            for i in tqdm(range(num)):
                medium = random.choice(MEDIUMS)
                subject = random.choice(SUBJECTS)
                pos_prompt = f"a {medium} of a {target_word} {subject}"
                neg_prompt = f"a {medium} of a {opposite} {subject}"
                pos_toks = self.pipe.tokenizer(pos_prompt, return_tensors="pt", padding="max_length", truncation=True,
                                          max_length=77).input_ids.cuda()
                neg_toks = self.pipe.tokenizer(neg_prompt, return_tensors="pt", padding="max_length", truncation=True,
                                          max_length=77).input_ids.cuda()
                pos = self.pipe.text_encoder(pos_toks).pooler_output
                neg = self.pipe.text_encoder(neg_toks).pooler_output
                positives.append(pos)
                negatives.append(neg)

        positives = torch.cat(positives, dim=0)
        negatives = torch.cat(negatives, dim=0)

        diffs = positives - negatives

        avg_diff = diffs.mean(0, keepdim=True)
        return avg_diff


    def generate(self,
        prompt = "a photo of a house",
        scale = 2,
        seed = 15,
        only_pooler = False,
        correlation_weight_factor = 1.0,
        ** pipeline_kwargs
        ):
        # if doing full sequence, [-0.3,0.3] work well, higher if correlation weighted is true
        # if pooler token only [-4,4] work well

        with torch.no_grad():
            toks = self.pipe.tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True,
                                  max_length=77).input_ids.cuda()
        prompt_embeds = self.pipe.text_encoder(toks).last_hidden_state

        if only_pooler:
            prompt_embeds[:, toks.argmax()] = prompt_embeds[:, toks.argmax()] + self.avg_diff * scale
        else:
            normed_prompt_embeds = prompt_embeds / prompt_embeds.norm(dim=-1, keepdim=True)
        sims = normed_prompt_embeds[0] @ normed_prompt_embeds[0].T
        weights = sims[toks.argmax(), :][None, :, None].repeat(1, 1, 768)

        standard_weights = torch.ones_like(weights)

        weights = standard_weights + (weights - standard_weights) * correlation_weight_factor

        # weights = torch.sigmoid((weights-0.5)*7)

        prompt_embeds = prompt_embeds + (weights * self.avg_diff[None, :].repeat(1, 77, 1) * scale)

        torch.manual_seed(seed)
        image = self.pipe(prompt_embeds=prompt_embeds, **pipeline_kwargs).images

        return image

    def spectrum(self,
                 prompt="a photo of a house",
                 low_scale=-2,
                 high_scale=2,
                 steps=5,
                 seed=15,
                 only_pooler=False,
                 correlation_weight_factor=1.0,
                 **pipeline_kwargs
                 ):

        images = []
        for i in range(steps):
            scale = low_scale + (high_scale - low_scale) * i / (steps - 1)
            image = self.generate(prompt, scale, seed, only_pooler, correlation_weight_factor, **pipeline_kwargs)
            images.append(image[0])

        canvas = Image.new('RGB', (640 * steps, 640))
        for i, im in enumerate(images):
            canvas.paste(im, (640 * i, 0))

        return canvas