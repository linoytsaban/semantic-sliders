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
            opposite: str,
            target_word_2nd: str = "",
            opposite_2nd: str = "",
            iterations: int = 300,
    ):

        self.device = device
        self.pipe = sd_pipe.to(self.device)
        self.iterations = iterations
        self.avg_diff = self.find_latent_direction(target_word, opposite)
        if target_word_2nd != "" or opposite_2nd != "":
            self.avg_diff_2nd = self.find_latent_direction(target_word_2nd, opposite_2nd)
        else:
            self.avg_diff_2nd = None


    def find_latent_direction(self,
                              target_word:str,
                              opposite:str):

        # lets identify a latent direction by taking differences between opposites
        # target_word = "happy"
        # opposite = "sad"


        with torch.no_grad():
            positives = []
            negatives = []
            for i in tqdm(range(self.iterations)):
                medium = random.choice(MEDIUMS)
                subject = random.choice(SUBJECTS)
                pos_prompt = f"a {medium} of a {target_word} {subject}"
                neg_prompt = f"a {medium} of a {opposite} {subject}"
                pos_toks = self.pipe.tokenizer(pos_prompt, return_tensors="pt", padding="max_length", truncation=True,
                                          max_length=self.pipe.tokenizer.model_max_length).input_ids.cuda()
                neg_toks = self.pipe.tokenizer(neg_prompt, return_tensors="pt", padding="max_length", truncation=True,
                                          max_length=self.pipe.tokenizer.model_max_length).input_ids.cuda()
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
        scale = 2.,
        scale_2nd = 0., # scale for the 2nd dim directions when avg_diff_2nd is not None
        seed = 15,
        only_pooler = False,
        normalize_scales = False, # whether to normalize the scales when avg_diff_2nd is not None
        correlation_weight_factor = 1.0,
        **pipeline_kwargs
        ):
        # if doing full sequence, [-0.3,0.3] work well, higher if correlation weighted is true
        # if pooler token only [-4,4] work well

        with torch.no_grad():
            toks = self.pipe.tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True,
                                  max_length=self.pipe.tokenizer.model_max_length).input_ids.cuda()
        prompt_embeds = self.pipe.text_encoder(toks).last_hidden_state

        if self.avg_diff_2nd and normalize_scales:
            denominator = abs(scale) + abs(scale_2nd)
            scale = scale / denominator
            scale_2nd = scale_2nd / denominator
        if only_pooler:
            prompt_embeds[:, toks.argmax()] = prompt_embeds[:, toks.argmax()] + self.avg_diff * scale
            if self.avg_diff_2nd:
                prompt_embeds[:, toks.argmax()] += self.avg_diff_2nd * scale_2nd
        else:
            normed_prompt_embeds = prompt_embeds / prompt_embeds.norm(dim=-1, keepdim=True)
        sims = normed_prompt_embeds[0] @ normed_prompt_embeds[0].T
        weights = sims[toks.argmax(), :][None, :, None].repeat(1, 1, 768)

        standard_weights = torch.ones_like(weights)

        weights = standard_weights + (weights - standard_weights) * correlation_weight_factor

        # weights = torch.sigmoid((weights-0.5)*7)
        prompt_embeds = prompt_embeds + (
                    weights * self.avg_diff[None, :].repeat(1, self.pipe.tokenizer.model_max_length, 1) * scale)
        if self.avg_diff_2nd:
            prompt_embeds += weights * self.avg_diff_2nd[None, :].repeat(1, self.pipe.tokenizer.model_max_length, 1) * scale_2nd


        torch.manual_seed(seed)
        image = self.pipe(prompt_embeds=prompt_embeds, **pipeline_kwargs).images

        return image

    def spectrum(self,
                 prompt="a photo of a house",
                 low_scale=-2,
                 low_scale_2nd=-2,
                 high_scale=2,
                 high_scale_2nd=2,
                 steps=5,
                 seed=15,
                 only_pooler=False,
                 normalize_scales=False,
                 correlation_weight_factor=1.0,
                 **pipeline_kwargs
                 ):

        images = []
        for i in range(steps):
            scale = low_scale + (high_scale - low_scale) * i / (steps - 1)
            scale_2nd = low_scale_2nd + (high_scale_2nd - low_scale_2nd) * i / (steps - 1)
            image = self.generate(prompt, scale, scale_2nd, seed, only_pooler, normalize_scales, correlation_weight_factor, **pipeline_kwargs)
            images.append(image[0])

        canvas = Image.new('RGB', (640 * steps, 640))
        for i, im in enumerate(images):
            canvas.paste(im, (640 * i, 0))

        return canvas

class CLIPSliderXL(CLIPSlider):

    def find_latent_direction(self,
                              target_word:str,
                              opposite:str):

        # lets identify a latent direction by taking differences between opposites
        # target_word = "happy"
        # opposite = "sad"


        with torch.no_grad():
            positives = []
            negatives = []
            positives2 = []
            negatives2 = []
            for i in tqdm(range(self.iterations)):
                medium = random.choice(MEDIUMS)
                subject = random.choice(SUBJECTS)
                pos_prompt = f"a {medium} of a {target_word} {subject}"
                neg_prompt = f"a {medium} of a {opposite} {subject}"

                pos_toks = self.pipe.tokenizer(pos_prompt, return_tensors="pt", padding="max_length", truncation=True,
                                          max_length=self.pipe.tokenizer.model_max_length).input_ids.cuda()
                neg_toks = self.pipe.tokenizer(neg_prompt, return_tensors="pt", padding="max_length", truncation=True,
                                          max_length=self.pipe.tokenizer.model_max_length).input_ids.cuda()
                pos = self.pipe.text_encoder(pos_toks).pooler_output
                neg = self.pipe.text_encoder(neg_toks).pooler_output
                positives.append(pos)
                negatives.append(neg)

                pos_toks2 = self.pipe.tokenizer_2(pos_prompt, return_tensors="pt", padding="max_length", truncation=True,
                                             max_length=self.pipe.tokenizer_2.model_max_length).input_ids.cuda()
                neg_toks2 = self.pipe.tokenizer_2(neg_prompt, return_tensors="pt", padding="max_length", truncation=True,
                                             max_length=self.pipe.tokenizer_2.model_max_length).input_ids.cuda()
                pos2 = self.pipe.text_encoder_2(pos_toks2).text_embeds
                neg2 = self.pipe.text_encoder_2(neg_toks2).text_embeds
                positives2.append(pos2)
                negatives2.append(neg2)

        positives = torch.cat(positives, dim=0)
        negatives = torch.cat(negatives, dim=0)
        diffs = positives - negatives
        avg_diff = diffs.mean(0, keepdim=True)

        positives2 = torch.cat(positives2, dim=0)
        negatives2 = torch.cat(negatives2, dim=0)
        diffs2 = positives2 - negatives2
        avg_diff2 = diffs2.mean(0, keepdim=True)
        return (avg_diff, avg_diff2)

    def generate(self,
        prompt = "a photo of a house",
        scale = 2,
        scale_2nd = 2,
        seed = 15,
        only_pooler = False,
        normalize_scales = False,
        correlation_weight_factor = 1.0,
        **pipeline_kwargs
        ):
        # if doing full sequence, [-0.3,0.3] work well, higher if correlation weighted is true
        # if pooler token only [-4,4] work well

        text_encoders = [self.pipe.text_encoder, self.pipe.text_encoder_2]
        tokenizers = [self.pipe.tokenizer, self.pipe.tokenizer_2]
        with torch.no_grad():
            # toks = pipe.tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=77).input_ids.cuda()
            # prompt_embeds = pipe.text_encoder(toks).last_hidden_state

            prompt_embeds_list = []

            for i, text_encoder in enumerate(text_encoders):

                tokenizer = tokenizers[i]
                text_inputs = tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                toks = text_inputs.input_ids

                prompt_embeds = text_encoder(
                    toks.to(text_encoder.device),
                    output_hidden_states=True,
                )

                # We are only ALWAYS interested in the pooled output of the final text encoder
                pooled_prompt_embeds = prompt_embeds[0]
                prompt_embeds = prompt_embeds.hidden_states[-2]

                if self.avg_diff_2nd and normalize_scales:
                    denominator = abs(scale) + abs(scale_2nd)
                    scale = scale / denominator
                    scale_2nd = scale_2nd / denominator
                if only_pooler:
                    prompt_embeds[:, toks.argmax()] = prompt_embeds[:, toks.argmax()] + self.avg_diff[0] * scale
                    if self.avg_diff_2nd:
                        prompt_embeds[:, toks.argmax()] += self.avg_diff_2nd[0] * scale_2nd
                else:
                    normed_prompt_embeds = prompt_embeds / prompt_embeds.norm(dim=-1, keepdim=True)
                    sims = normed_prompt_embeds[0] @ normed_prompt_embeds[0].T

                    if i == 0:
                        weights = sims[toks.argmax(), :][None, :, None].repeat(1, 1, 768)

                        standard_weights = torch.ones_like(weights)

                        weights = standard_weights + (weights - standard_weights) * correlation_weight_factor
                        prompt_embeds = prompt_embeds + (weights * self.avg_diff[0][None, :].repeat(1, self.pipe.tokenizer.model_max_length, 1) * scale)
                        if self.avg_diff_2nd:
                            prompt_embeds += (weights * self.avg_diff_2nd[0][None, :].repeat(1, self.pipe.tokenizer.model_max_length, 1) * scale_2nd)
                    else:
                        weights = sims[toks.argmax(), :][None, :, None].repeat(1, 1, 1280)

                        standard_weights = torch.ones_like(weights)

                        weights = standard_weights + (weights - standard_weights) * correlation_weight_factor
                        prompt_embeds = prompt_embeds + (weights * self.avg_diff[1][None, :].repeat(1, self.pipe.tokenizer_2.model_max_length, 1) * scale)
                        if self.avg_diff_2nd:
                            prompt_embeds += (weights * self.avg_diff_2nd[1][None, :].repeat(1, self.pipe.tokenizer_2.model_max_length, 1) * scale_2nd)

                bs_embed, seq_len, _ = prompt_embeds.shape
                prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
                prompt_embeds_list.append(prompt_embeds)

            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
            pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)

            torch.manual_seed(seed)
            image = self.pipe(prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds,
                         **pipeline_kwargs).images

        return image


class CLIPSlider3(CLIPSlider):
    def find_latent_direction(self,
                              target_word:str,
                              opposite:str):

        # lets identify a latent direction by taking differences between opposites
        # target_word = "happy"
        # opposite = "sad"


        with torch.no_grad():
            positives = []
            negatives = []
            positives2 = []
            negatives2 = []
            for i in tqdm(range(self.iterations)):
                medium = random.choice(MEDIUMS)
                subject = random.choice(SUBJECTS)
                pos_prompt = f"a {medium} of a {target_word} {subject}"
                neg_prompt = f"a {medium} of a {opposite} {subject}"

                pos_toks = self.pipe.tokenizer(pos_prompt, return_tensors="pt", padding="max_length", truncation=True,
                                          max_length=self.pipe.tokenizer.model_max_length).input_ids.cuda()
                neg_toks = self.pipe.tokenizer(neg_prompt, return_tensors="pt", padding="max_length", truncation=True,
                                          max_length=self.pipe.tokenizer.model_max_length).input_ids.cuda()
                pos = self.pipe.text_encoder(pos_toks).text_embeds
                neg = self.pipe.text_encoder(neg_toks).text_embeds
                positives.append(pos)
                negatives.append(neg)

                pos_toks2 = self.pipe.tokenizer_2(pos_prompt, return_tensors="pt", padding="max_length", truncation=True,
                                             max_length=self.pipe.tokenizer_2.model_max_length).input_ids.cuda()
                neg_toks2 = self.pipe.tokenizer_2(neg_prompt, return_tensors="pt", padding="max_length", truncation=True,
                                             max_length=self.pipe.tokenizer_2.model_max_length).input_ids.cuda()
                pos2 = self.pipe.text_encoder_2(pos_toks2).text_embeds
                neg2 = self.pipe.text_encoder_2(neg_toks2).text_embeds
                positives2.append(pos2)
                negatives2.append(neg2)

        positives = torch.cat(positives, dim=0)
        negatives = torch.cat(negatives, dim=0)
        diffs = positives - negatives
        avg_diff = diffs.mean(0, keepdim=True)

        positives2 = torch.cat(positives2, dim=0)
        negatives2 = torch.cat(negatives2, dim=0)
        diffs2 = positives2 - negatives2
        avg_diff2 = diffs2.mean(0, keepdim=True)
        return (avg_diff, avg_diff2)

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
        clip_text_encoders = [self.pipe.text_encoder, self.pipe.text_encoder_2]
        clip_tokenizers = [self.pipe.tokenizer, self.pipe.tokenizer_2]
        with torch.no_grad():
            # toks = pipe.tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=77).input_ids.cuda()
            # prompt_embeds = pipe.text_encoder(toks).last_hidden_state

            clip_prompt_embeds_list = []
            clip_pooled_prompt_embeds_list = []
            for i, text_encoder in enumerate(clip_text_encoders):

                if i < 2:
                    tokenizer = clip_tokenizers[i]
                    text_inputs = tokenizer(
                        prompt,
                        padding="max_length",
                        max_length=tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    )
                    toks = text_inputs.input_ids

                    prompt_embeds = text_encoder(
                        toks.to(text_encoder.device),
                        output_hidden_states=True,
                    )

                    # We are only ALWAYS interested in the pooled output of the final text encoder
                    pooled_prompt_embeds = prompt_embeds[0]
                    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
                    clip_pooled_prompt_embeds_list.append(pooled_prompt_embeds)
                    prompt_embeds = prompt_embeds.hidden_states[-2]
                else:
                    text_inputs = self.pipe.tokenizer_3(
                        prompt,
                        padding="max_length",
                        max_length=self.tokenizer_max_length,
                        truncation=True,
                        add_special_tokens=True,
                        return_tensors="pt",
                    )
                    toks = text_inputs.input_ids
                    prompt_embeds = self.pipe.text_encoder_3(toks.to(self.device))[0]
                    t5_prompt_embed_shape = prompt_embeds.shape[-1]

                if only_pooler:
                    prompt_embeds[:, toks.argmax()] = prompt_embeds[:, toks.argmax()] + self.avg_diff[0] * scale
                else:
                    normed_prompt_embeds = prompt_embeds / prompt_embeds.norm(dim=-1, keepdim=True)
                    sims = normed_prompt_embeds[0] @ normed_prompt_embeds[0].T
                    if i == 0:
                        weights = sims[toks.argmax(), :][None, :, None].repeat(1, 1, 768)

                        standard_weights = torch.ones_like(weights)

                        weights = standard_weights + (weights - standard_weights) * correlation_weight_factor
                        prompt_embeds = prompt_embeds + (weights * self.avg_diff[0][None, :].repeat(1, self.pipe.tokenizer.model_max_length, 1) * scale)
                    else:
                        weights = sims[toks.argmax(), :][None, :, None].repeat(1, 1, 1280)

                        standard_weights = torch.ones_like(weights)

                        weights = standard_weights + (weights - standard_weights) * correlation_weight_factor
                        prompt_embeds = prompt_embeds + (weights * self.avg_diff[1][None, :].repeat(1, self.pipe.tokenizer_2.model_max_length, 1) * scale)

                bs_embed, seq_len, _ = prompt_embeds.shape
                prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
                if i < 2:
                    clip_prompt_embeds_list.append(prompt_embeds)

            clip_prompt_embeds = torch.concat(clip_prompt_embeds_list, dim=-1)
            clip_pooled_prompt_embeds = torch.concat(clip_pooled_prompt_embeds_list, dim=-1)

            clip_prompt_embeds = torch.nn.functional.pad(
                clip_prompt_embeds, (0, t5_prompt_embed_shape - clip_prompt_embeds.shape[-1])
            )

            prompt_embeds = torch.cat([clip_prompt_embeds, prompt_embeds], dim=-2)



            torch.manual_seed(seed)
            image = self.pipe(prompt_embeds=prompt_embeds, pooled_prompt_embeds=clip_pooled_prompt_embeds,
                         **pipeline_kwargs).images

        return image