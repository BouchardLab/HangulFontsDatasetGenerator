import os, glob, sys

folder = sys.argv[1]
print(folder)

names = [('JoseonIlboMyeongjo', '┴╢╝▒└╧║╕╕э┴╢'),
         ('NanumMyeongjoEB', '│к┤о╕э┴╢EB'),
         ('NanumSongeulssibut', '│к┤о╝╒▒█╛╛║╫'),
         ('SeoulHangangB', '╝н┐я╟╤░н └х├╝B'),
         ('SeoulHangangBL', '╝н┐я╟╤░н └х├╝BL'),
         ('SeoulHangangEB', '╝н┐я╟╤░н └х├╝EB'),
         ('SeoulHangangL', '╝н┐я╟╤░н └х├╝L'),
         ('SeoulHangangM', '╝н┐я╟╤░н └х├╝M')]

for new_name, old_name in names:
    for dirpath, dirnames, filenames in os.walk(folder):
        for fname in filenames:
            if old_name in fname:
                old_path = os.path.join(dirpath, fname)
                new_path = os.path.join(dirpath, fname.replace(old_name, new_name))
                print(old_path)
                print(new_path)
                os.rename(old_path, new_path)

for new_name, old_name in names:
    for dirpath, dirnames, filenames in os.walk(folder):
        for dirname in dirnames:
            if old_name in dirname:
                old_path = os.path.join(dirpath, dirname)
                new_path = os.path.join(dirpath, dirname.replace(old_name, new_name))
                print(old_path)
                print(new_path)
                os.rename(old_path, new_path)
