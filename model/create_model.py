import create_model_retinanet
import create_model_centernet
from data_utils.data import BrailleDataset

def create_model(params, settings):
    if params.model == 'retina':
        dataset_class = BrailleDataset
        model, collate_fn, loss = create_model_retinanet.create_model_retinanet(params, device=settings.device)
    elif params.model == 'centernet':
        model, dataset_class, collate_fn, loss = create_model_centernet.create_model_centernet(params, device=settings.device)
    else:
        raise Exception(f'incorrect params.model: {params.model}')
    return model, dataset_class, collate_fn, loss