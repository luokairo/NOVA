import json

import cv2
import torch
import traceback
torch._dynamo.config.cache_size_limit = 64

from tools.run_infinity import *


def process_short_text(short_text):
    if '--' in short_text:
        processed_text = short_text.split('--')[0]
        if processed_text:
            short_text = processed_text
    return short_text

np.random.seed(0)

lines4infer = []
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    parser.add_argument('--rewrite_prompt', type=int, default=0, choices=[0,1])
    parser.add_argument('--out_dir', type=str, default='')
    parser.add_argument('--coco30k_prompts', type=int, default=0, choices=[0,1])
    parser.add_argument('--save4fid_eval', type=int, default=0, choices=[0,1])
    parser.add_argument('--save_recons_img', type=int, default=0, choices=[0,1])
    parser.add_argument('--jsonl_filepath', type=str, default='')
    parser.add_argument('--long_caption_fid', type=int, default=1, choices=[0,1])
    parser.add_argument('--fid_max_examples', type=int, default=-1)
    parser.add_argument('--n_samples', type=int, default=1)
    args = parser.parse_args()
    if not args.coco30k_prompts and not args.jsonl_filepath:
        parser.error('Provide --jsonl_filepath or --coco30k_prompts 1.')
    args.cfg = list(map(float, args.cfg.split(',')))
    if len(args.cfg) == 1:
        args.cfg = args.cfg[0]
        
    if args.coco30k_prompts:
        from T2IBenchmark.datasets import get_coco_30k_captions, get_coco_fid_stats
        id2caption = get_coco_30k_captions()
        captions = []
        ids = []
        lines4infer = []
        for d in id2caption.items():
            ids.append(d[0])
            captions.append(d[1])
        np.random.shuffle(captions)
        lines4infer = [{'prompt': prompt, 'h_div_w': 1.0, 'infer_type': 'infer/coco30k_prompt'} for prompt in captions]
    
    if args.jsonl_filepath:
        lines4infer = []
        with open(args.jsonl_filepath, 'r') as f:
            cnt = 0
            for line in f:
                meta = json.loads(line)
                gt_image_path = meta['image_path']
                assert osp.exists(gt_image_path), gt_image_path
                if args.long_caption_fid:
                    prompt = meta['long_caption']
                else:
                    prompt = meta['text']
                if not prompt:
                    continue
                lines4infer.append({
                    'prompt': prompt,
                    'h_div_w': meta['h_div_w'],
                    'infer_type': 'val/laion_coco_long_caption',
                    'gt_image_path': gt_image_path,
                    'meta_line': line,
                })

    if args.fid_max_examples > 0:
        lines4infer = lines4infer[:args.fid_max_examples]
    print(f'Totally {len(lines4infer)} items for infer')

    if args.out_dir:
        out_dir = args.out_dir
    else:
        out_dir = osp.join('output', osp.basename(osp.dirname(args.model_path)), osp.splitext(osp.basename(args.model_path))[0], f'coco30k_infer' if args.coco30k_prompts else 'comprehensive_infer')
    print(f'save to {out_dir}')

    # load text encoder
    text_tokenizer, text_encoder = load_tokenizer(t5_path =args.text_encoder_ckpt)
    # load vae
    vae = load_visual_tokenizer(args)
    # load infinity
    infinity = load_transformer(vae, args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # inference
    if osp.exists(out_dir):
        # shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        shutil.copyfile(__file__, osp.join(out_dir, osp.basename(__file__)))

    jsonl_list = []
    cnt = 0
    for i, infer_data in enumerate(lines4infer):
        try:
            prompt = infer_data['prompt']
            prompt = process_short_text(prompt)
            prompt_id = get_prompt_id(prompt)
            save_file = osp.join(out_dir, 'pred', f'{prompt_id}.jpg')
            if osp.exists(save_file):
                continue

            h_div_w_template = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - infer_data['h_div_w']))]
            scale_schedule = dynamic_resolution_h_w[h_div_w_template][args.pn]['scales']
            scale_schedule = [(1, h, w) for (t,h, w) in scale_schedule]
            if args.apply_spatial_patchify:
                vae_scale_schedule = [(pt, 2*ph, 2*pw) for pt, ph, pw in scale_schedule]
            else:
                vae_scale_schedule = scale_schedule
            tgt_h, tgt_w = dynamic_resolution_h_w[h_div_w_template][args.pn]['pixel']

            img_list = []
            gt_ls_Bl = []
            if ('gt_image_path' in infer_data):
                gt_img, recons_img, all_bit_indices = joint_vi_vae_encode_decode(vae, infer_data['gt_image_path'], vae_scale_schedule, device, tgt_h, tgt_w)
                gt_ls_Bl = all_bit_indices
            else:
                if args.save4fid_eval:
                    continue
            
            if args.coco30k_prompts or args.save4fid_eval:
                concate_img = gen_one_img(infinity, vae, text_tokenizer, text_encoder, prompt, g_seed=0, gt_leak=0, gt_ls_Bl=gt_ls_Bl, tau_list=args.tau, cfg_sc=3, cfg_list=args.cfg, scale_schedule=scale_schedule, cfg_insertion_layer=[args.cfg_insertion_layer], vae_type=args.vae_type, sampling_per_bits=args.sampling_per_bits)
            else:
                g_seed = 0 if args.n_samples == 1 else None
                tmp_img_list = []
                for _ in range(args.n_samples):
                    tmp_img_list.append(gen_one_img(infinity, vae, text_tokenizer, text_encoder, prompt, g_seed=g_seed, gt_leak=0, gt_ls_Bl=gt_ls_Bl, tau_list=args.tau, cfg_sc=3, cfg_list=args.cfg, scale_schedule=scale_schedule, cfg_insertion_layer=[args.cfg_insertion_layer], vae_type=args.vae_type, sampling_per_bits=1, top_k=0))
                img_list.append(np.concatenate(tmp_img_list, axis=1))

                tmp_img_list = []
                for _ in range(args.n_samples):
                    tmp_img_list.append(gen_one_img(infinity, vae, text_tokenizer, text_encoder, prompt, g_seed=g_seed, gt_leak=0, gt_ls_Bl=gt_ls_Bl, tau_list=args.tau, cfg_sc=3, cfg_list=args.cfg, scale_schedule=scale_schedule, cfg_insertion_layer=[args.cfg_insertion_layer], vae_type=args.vae_type, sampling_per_bits=8, top_k=0))
                img_list.append(np.concatenate(tmp_img_list, axis=1))

                tmp_img_list = []
                for _ in range(args.n_samples):
                    tmp_img_list.append(gen_one_img(infinity, vae, text_tokenizer, text_encoder, prompt, g_seed=g_seed, gt_leak=0, gt_ls_Bl=gt_ls_Bl, tau_list=args.tau, cfg_sc=3, cfg_list=args.cfg, scale_schedule=scale_schedule, cfg_insertion_layer=[args.cfg_insertion_layer], vae_type=args.vae_type, sampling_per_bits=16, top_k=0))
                img_list.append(np.concatenate(tmp_img_list, axis=1))
                
                if args.n_samples == 1:
                    concate_img = np.concatenate([np.array(item) for item in img_list], 1)
                else:
                    concate_img = np.concatenate([np.array(item) for item in img_list], 0)
                concate_img = Image.fromarray(concate_img)
            
            os.makedirs(osp.dirname(save_file), exist_ok=True)
            cv2.imwrite(save_file, concate_img.cpu().numpy())
            infer_data['image_path'] = osp.abspath(save_file)

            if args.save4fid_eval:
                save_file = osp.join(out_dir, 'gt', f'{prompt_id}.jpg')
                os.makedirs(osp.dirname(save_file), exist_ok=True)
                gt_img = Image.fromarray(np.array(gt_img))
                assert not osp.exists(save_file), f'{save_file} exists, infer_data: {infer_data}'
                gt_img.save(save_file)
                if args.save_recons_img:
                    save_file = osp.join(out_dir, 'recons', f'{prompt_id}.jpg')
                    os.makedirs(osp.dirname(save_file), exist_ok=True)
                    recons_img = Image.fromarray(np.array(recons_img))
                    recons_img.save(save_file)

            jsonl_list.append(json.dumps(infer_data)+'\n')
            jsonl_file = osp.join(out_dir, 'meta_info.jsonl')
            with open(jsonl_file, 'w') as f:
                f.writelines(jsonl_list)
            print(f'Save to {osp.abspath(jsonl_file)}')
        except Exception as e:
            print(f"{e}", traceback.print_exc())
