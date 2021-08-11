
def snap(label: str):
    import os
    snapdir = "snapshots/" + label
    print("snapdir is %s" % snapdir)
    os.system("mkdir -p " + snapdir)
    os.system("cp *.py %s" % snapdir)
    os.system("cp *.ipynb %s" % snapdir)
    os.system("cp -rp elections util network %s" % snapdir)


