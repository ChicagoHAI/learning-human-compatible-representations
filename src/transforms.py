from torchvision import transforms

def get_transform(transform, aug=True):
    if transform == "bm":
        return bm_transform_aug() if aug else bm_transform()
    elif transform == "wv":
        return bird_transform()
    elif transform == "xray":
        return xray_transform_aug() if aug else xray_transform()
    elif transform == "bird":
        return bird_transform_aug() if aug else bird_transform()
    elif transform == "resn":
        return resn_normalize()
    elif transform == "wv_3d":
        return bird_transform()

def bm_transform_aug():
    affine = {}
    affine["degrees"] = 30 #hparams.rotate
    # if hparams.translate > 0: 
    #     translate = hparams.translate
    #     affine["translate"] = (translate, translate)
    if 0.2 > 0: 
        scale = 0.2
        affine["scale"] = (1 - scale, 1 + scale)
    # if hparams.shear > 0:
    #     shear = hparams.shear
    #     affine["shear"] = (-shear, shear, -shear, shear)
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomHorizontalFlip(0),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomAffine(**affine)
    ])
    return transform

def bm_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def no_transform():
    return transforms.Compose([
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def resn_normalize():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def resn_normalize_aug():
    return transforms.Compose([
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def bird_transform_aug():
    return transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def bird_transform():
    return transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def xray_transform_aug():
    return transforms.Compose([
        transforms.Resize((2500,2500)),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

def xray_transform():
    return transforms.Compose([
        transforms.Resize((2500,2500)),
        transforms.ToTensor(),
    ])

