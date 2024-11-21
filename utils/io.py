


def write_list2file(out_dir,failed_objs):
    with open(out_dir,'w') as fp:
        for item in failed_objs:
            fp.write(f"{item}\n")

        fp.close()