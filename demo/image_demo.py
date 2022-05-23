# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmcls.apis import inference_model, init_model, show_result_pyplot


def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cpu', help='Device used for inference')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    # test a single image
    import pdb
    # pdb.set_trace()
    result = inference_model(model, args.img)
    # show the results
    show_result_pyplot(model, args.img, result)


if __name__ == '__main__':
    main()
