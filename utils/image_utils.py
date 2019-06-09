from PIL import Image, ImageDraw
import os, random


class ImageHelper:
    def __init__(self, path):
        self._path = path

        if not os.path.exists(self._path):
            os.makedirs(self._path)

    def draw_lines(self, img_array, target_name, lines=[[]], colors=[], orientation='horizontal'):
        images = list(map(lambda x: Image.open(f"{self._path}/{x}"), img_array))
        widths, heights = zip(*[i.size for i in images])
        horizontal = True if orientation == 'horizontal' else False

        if horizontal:
            result = Image.new("RGBA", (sum(widths), max(heights)))
        else:
            result = Image.new("RGBA", (max(widths), sum(heights)))

        x = 0

        for i in images:
            result.paste(i, (x * horizontal, x * (not horizontal)))
            x += i.size[0] if horizontal else i.size[1]

        canvas = ImageDraw.Draw(result)

        if len(colors) < 1:
            colors = [
                (255, 0, 0, 255),
                (0, 0, 255, 255),
                (0, 255, 0, 255)
            ]

        if len(lines) > len(colors):
            for i in range(len(colors), len(lines)):
                colors.append((random.randint(255), random.randint(255), random.randint(255), 255))

        for x, point_pairs in enumerate(lines):
            for pair in point_pairs:
                for i, point in enumerate(pair):
                    if i > 0:
                        canvas.line((pair[i - 1]['x'], pair[i - 1]['y'],
                                     pair[i]['x'] + (images[i - 1].size[0] * horizontal),
                                     pair[i]['y'] + (images[i - 1].size[1] * (not horizontal))),
                                     fill=(colors[x][0] % 256, colors[x][1] % 256, colors[x][2] % 256, colors[x][3] % 256))

        del canvas

        result.save(f'{self._path}/{target_name}')
