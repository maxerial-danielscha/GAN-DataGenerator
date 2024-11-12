import numpy as np
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
import collections



def bandpass_filter(image, low_cutoff, high_cutoff):
            f_transform = np.fft.fft2(image)
            f_transform_shifted = np.fft.fftshift(f_transform)
            rows, cols = image.shape
            crow, ccol = rows // 2, cols // 2

            x = np.linspace(-ccol, ccol - 1, cols)
            y = np.linspace(-crow, crow - 1, rows)
            X, Y = np.meshgrid(x, y)
            radius = np.sqrt(X**2 + Y**2)

            bandpass_mask = np.zeros((rows, cols), dtype=np.float32)
            bandpass_mask[(radius >= low_cutoff) & (radius <= high_cutoff)] = 1
            filtered_f_transform = f_transform_shifted * bandpass_mask

            return np.fft.ifft2(np.fft.ifftshift(filtered_f_transform)).real
        
        
def bandpass_filter_test_params(image=None, low_low=0, low_high=20, high_low=20, high_high=80, show=False, save_path="test_bandpass.png"):
    """Utility to test the bandpass_filter with different parameters
    Performs grid search over specified range and plots it for visual assesment

    Args:
        image (_type_, optional): _description_. Defaults to None.
        low_low (int, optional): _description_. Defaults to 0.
        low_high (int, optional): _description_. Defaults to 20.
        high_low (int, optional): _description_. Defaults to 20.
        high_high (int, optional): _description_. Defaults to 80.
    """
    images = []
    titles = []
    for low  in tqdm(np.linspace(0,20,10)):
        for high in np.linspace(20,100,10):
            low = int(low)
            high = int(high)
            cropped_image = bandpass_filter(image, low_cutoff=low, high_cutoff=high)
            images.append(cropped_image.copy())
            titles.append(f"l:{low}, h:{high}")

    plot_multi(images=images,
                        titles=titles, 
                        img_per_row=10,
                        figsize_of_images=(4,4),
                        gray=True,
                        main_title="Test bandpass params. low: [{low  }]",
                        show=show,
                        save_path=save_path)
    
        
        
        
def plot_multi(images: list, gray=False, titles=[], img_per_row=2, main_title="", save_path=None, show=True, figsize_of_images= (4,4)):
    """Utility to plot an arbitrary number of images in one plot, with easy settings.
    

    Args:
        images (list): _description_
        gray (bool, optional): _description_. Defaults to False.
        titles (list, optional): titles of the individual images, titles[i] will be the title of images[i]. Defaults to [].
        img_per_row (int, optional): _description_. Defaults to 2.
        main_title (str, optional): Supertitle of whole plot. Defaults to "".
        save_path (_type_, optional): where to save the figure. Defaults to None.
        show (bool, optional): if the figure should be shown. Defaults to True.
        figsize_of_images (tuple, optional): size of the individual images, overal figure will be made to accommodate that size. Defaults to (4,4).
    """
    
    count = len(images)
    cols = int(img_per_row)
    rows = int(np.ceil(count/cols))

    figsize = (figsize_of_images[0] * cols, figsize_of_images[1] * rows + 4)
    plt.figure(figsize=figsize)
    
    if main_title != "":
        plt.suptitle(main_title)        

    for index, img in enumerate(images):
        plt.subplot(rows,cols,index+1)        
        plt.xticks([])
        plt.yticks([])
        if gray:
            # if len(img.shape) == 2:
            plt.gray()
        if len(titles) == count:
            plt.title(titles[index])
            
        plt.imshow(img)
        
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
        
    if show:
        plt.show()
    else:
        plt.close()
        
        
def plot(img, gray=False, title=""):
    """
    Utility to plot a single image

    Args:
        img (array): the image to be plotted
        gray (bool, optional): If True, cmap="gray". Defaults to False.
        title (str, optional): Plot title. Defaults to "".
    """    
              
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    if gray:
        plt.gray()
    if title != "":
        plt.title(title)
    plt.imshow(img)
    
        
def visualize_2D_scatter(X,y):
    pca = PCA(n_components=2)
    res = pca.fit_transform(X)
    plt.scatter(x=res[:,0], y=res[:,1], c=y)
    plt.show()




