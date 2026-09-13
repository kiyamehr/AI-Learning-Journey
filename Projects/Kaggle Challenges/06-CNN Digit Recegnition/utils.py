def scale_img(*images):
    return [image / 255.0 for image in images]
