c2f_scale=[101,301,501,701,901]
max_reso=128


for epoch in c2f_scale:
    # epoch = epoch -1
    new_reso = int(max_reso / (2 ** (len(c2f_scale) - c2f_scale.index(epoch) - 1)))


    print(f"epoch: {epoch}, new_reso: {new_reso}")