import os
import shutil
from os.path import join

def generate_html(figures_dir, output_dir="blurred-html", nc='32'):
    countries = set()
    ranges = set()
    pt_ft = set()

    output_dir = os.path.abspath(output_dir)
    figures_dir = os.path.abspath(figures_dir)
    os.chdir(figures_dir)
    figures_dir_ = os.path.split(os.getcwd())[1]
    os.chdir('..')
    for root, dirs, files in os.walk(figures_dir):
        path_parts = root.split(os.sep)
        if path_parts[-1] == 'clusters':
            for file in files:
                if file.endswith(".png"):
                    pt_ft.add(path_parts[-3])
                    range_part = path_parts[-2]
                    country_part = file.split('__')[0]
                    ranges.add(range_part)
                    countries.add(country_part)

    os.makedirs(output_dir, exist_ok=True)
    country_radios="\n".join([
        f'<label><input type="radio" name="country" value="{country}" onchange="updateImage()" {"checked" if country == sorted(countries)[0] else ""}>{country}</label>'
        for country in sorted(countries)
    ])
    range_radios="\n".join([
        f'<label><input type="radio" name="range" value="{range_}" onchange="updateImage()" {"checked" if range_ == sorted(ranges)[0] else ""}>{range_}</label>'
        for range_ in sorted(ranges)
    ])
    ft_radios="\n".join([
        f'<label><input type="radio" name="pt_ft" value="{range_}" onchange="updateImage()" {"checked" if range_ == sorted(pt_ft)[0] else ""}>{range_}</label>'
        for range_ in sorted(pt_ft)
    ])

    html = """
<!DOCTYPE html>
<html>
<head>
    <script>
        function updateImage() {
            var country = document.querySelector('input[name="country"]:checked').value;
            var range = document.querySelector('input[name="range"]:checked').value;
            var pt_ft = document.querySelector('input[name="pt_ft"]:checked').value;
            var imagePath = `"""
            
    html = html + figures_dir_
    html = html + """/${pt_ft}/${range}/clusters/${country}__hard_limit_20__top_k_""" + str(nc) + """__min_im_6_ranked.png`;
            document.getElementById('image').src = imagePath;
        }
        updateImage();
    </script>
</head>
<body>
    <h1>Image Viewer</h1>
    <div>
        <h2>Category:</h2>
        """
    html = html + country_radios + """
    </div>
    <div>
        <h2>Model:</h2>
        """
    html = html + ft_radios + """
    </div>
    <div>
        <h2>t_min-t_max:</h2>
        """
    html = html + range_radios + """
    </div>
    <br>
    <img id="image" src="" alt="Selected Image">
</body>
</html>
    """
    # Create an input directory for the images
    shutil.copytree(figures_dir_, join(output_dir, figures_dir_), dirs_exist_ok=True)
    with open(os.path.join(output_dir, "index.html"), "w") as file:
        file.write(html)

# Usage example
import sys
generate_html(sys.argv[1], sys.argv[2], (sys.argv[3] if len(sys.argv) == 4 else '32'))
