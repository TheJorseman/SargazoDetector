import torch
from torchvision import transforms
from PIL import Image
from cv2 import imwrite, cvtColor, COLOR_BGR2HSV
import numpy as np
from deeplabv3.model import get_model
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    #model = get_model()
    #model.load_state_dict("experiment/wights.pt")
    #model = torch.load("experiment/weights.pt").to(device)
    model = torch.load("checkpoint-50.pt").to(device)
    model.eval()
    data_transforms = transforms.Compose([transforms.Resize((512,384)), transforms.ToTensor()])
    name = "1622129640.Thu.May.27_15_34_00.GMT.2021.morelos.c2.timex.png"
    masked_name = "1622129640.Thu.May.27_15_34_00.GMT.2021.morelos.c2.timex-mask.png"
    test = Image.open("dataset/Images/" + name)
    mask = Image.open("dataset/BinaryMasks/" + masked_name)
    test_img = data_transforms(test).unsqueeze(0).to(device)
    with torch.no_grad() as nograd:
        output = model(test_img)
    img_out = output['out'].squeeze().transpose(1,0).detach().cpu().numpy()
    output_path = "experiment/" + name + "-2.png"
    plt.figure(figsize=(10,10));
    plt.subplot(131);
    plt.imshow(test);
    plt.title('Image')
    plt.axis('off');
    plt.subplot(132);
    plt.imshow(mask);
    plt.title('Ground Truth')
    plt.axis('off');
    plt.subplot(133);
    plt.imshow(output['out'].squeeze().transpose(1,0).cpu().detach().numpy());
    plt.title('Segmentation Output')
    plt.axis('off');
    plt.savefig('./experiment/' + name, bbox_inches='tight')
    #import pdb;pdb.set_trace()
    imwrite(output_path, (np.abs(img_out)*255).astype(np.uint8))
    


if __name__ == "__main__":
    main()