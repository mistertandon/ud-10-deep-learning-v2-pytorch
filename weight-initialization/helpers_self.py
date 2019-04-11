import numpy as np

def imshow(plt_axis, image, title, normalized=True):

    # Image Un-normalized process
    image = image.numpy().transpose((1, 2, 0))

    if normalized:
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.5, 0.5, 0.5])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    plt_axis.imshow(image)
    plt_axis.set_title(title, fontdict={'fontsize': 30})
    plt_axis.spines['top'].set_visible(False)
    plt_axis.spines['left'].set_visible(False)
    plt_axis.spines['bottom'].set_visible(False)
    plt_axis.spines['right'].set_visible(False)
    plt_axis.tick_params(axis='both', length=0)
    plt_axis.set_xticklabels('')
    plt_axis.set_yticklabels('')

    return plt_axis

def get_hello():
    print('hello')