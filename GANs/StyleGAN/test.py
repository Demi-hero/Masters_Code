import stylegan

test = stylegan.WGAN(lr = 0.0003, silent = False)
test.load(129)

a = test.img_generator()

print(a.shape)