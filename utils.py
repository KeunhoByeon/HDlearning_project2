import torch


# Save model state dict
def save_model(model, save_path):
    state_dict = {}
    state_dict['model'] = model.model
    state_dict['classes'] = model.classes
    state_dict['dim'] = model.dim

    state_dict['encoder'] = {}
    state_dict['encoder']['dim'] = model.encoder.dim
    state_dict['encoder']['features'] = model.encoder.features
    state_dict['encoder']['base'] = model.encoder.base
    state_dict['encoder']['basis'] = model.encoder.basis

    torch.save(state_dict, save_path)


def save_model_linear(model, save_path):
    state_dict = {}
    state_dict['model'] = model.model
    state_dict['classes'] = model.classes
    state_dict['dim'] = model.dim

    state_dict['encoder'] = {}
    state_dict['encoder']['dim'] = model.encoder.dim
    state_dict['encoder']['features'] = model.encoder.features
    state_dict['encoder']['m'] = model.encoder.m
    state_dict['encoder']['level'] = model.encoder.level
    state_dict['encoder']['levels'] = model.encoder.levels
    state_dict['encoder']['basis'] = model.encoder.basis

    torch.save(state_dict, save_path)


# Load trained model
def load_model(model, load_path):
    state_dict = torch.load(load_path)

    model.model = state_dict['model']
    model.classes = state_dict['classes']
    model.dim = state_dict['dim']

    model.encoder.dim = state_dict['encoder']['dim']
    model.encoder.features = state_dict['encoder']['features']
    model.encoder.base = state_dict['encoder']['base']
    model.encoder.basis = state_dict['encoder']['basis']


def load_model_linear(model, load_path):
    state_dict = torch.load(load_path)

    model.model = state_dict['model']
    model.classes = state_dict['classes']
    model.dim = state_dict['dim']

    model.encoder.dim = state_dict['encoder']['dim']
    model.encoder.features = state_dict['encoder']['features']
    model.encoder.m = state_dict['encoder']['m']
    model.encoder.level = state_dict['encoder']['level']
    model.encoder.levels = state_dict['encoder']['levels']
    model.encoder.basis = state_dict['encoder']['basis']
