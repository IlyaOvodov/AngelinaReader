from pathlib import Path
import sys

def create_model(params, device):
    if params.model == 'retina':
        import create_model_retinanet
        from data_utils.data import BrailleDataset
        dataset_class = BrailleDataset
        model, collate_fn, loss = create_model_retinanet.create_model_retinanet(params, device=device)
    elif params.model == 'centernet':
        # CenterNet project
        sys.path.insert(0, str(Path(params.model_params.center_net_path) / 'src' /'lib'))
        import create_model_centernet
        model, dataset_class, collate_fn, loss = create_model_centernet.create_model_centernet(params, device=device)
        sys.path.pop(0)
    else:
        raise Exception(f'incorrect params.model: {params.model}')
    return model, dataset_class, collate_fn, loss

def create_decoder(params):
    if params.model == 'retina':
        from model.my_decoder import RetinaNetDecoder
        decoder = RetinaNetDecoder(params)
    elif params.model == 'centernet':
        # CenterNet project
        sys.path.insert(0, str(Path(params.model_params.center_net_path) / 'src' /'lib'))
        import create_model_centernet
        decoder = create_model_centernet.CenterNetDecoder(params)
        sys.path.pop(0)
    else:
        raise Exception(f'incorrect params.model: {params.model}')
    return decoder
    