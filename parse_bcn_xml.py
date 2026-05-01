import xml.etree.ElementTree as ET
import re

def get_text(elem, ns):
    texts = []
    # Direct Texto child
    t = elem.find('ns:Texto', ns)
    if t is not None and t.text:
        texts.append(t.text.strip())
    # Recurse into nested EstructurasFuncionales
    sub = elem.find('ns:EstructurasFuncionales', ns)
    if sub is not None:
        for child in sub.findall('ns:EstructuraFuncional', ns):
            child_texts = get_text(child, ns)
            texts.extend(child_texts)
    return texts

def main():
    ns = {'ns': 'http://www.leychile.cl/esquemas'}
    tree = ET.parse('procedimiento_penal.xml')
    root = tree.getroot()

    lines = []

    # Header
    encabezado = root.find('ns:Encabezado', ns)
    if encabezado is not None:
        t = encabezado.find('ns:Texto', ns)
        if t is not None and t.text:
            lines.append(t.text.strip())
            lines.append('')

    # Main body
    estructuras = root.find('ns:EstructurasFuncionales', ns)
    if estructuras is not None:
        for child in estructuras.findall('ns:EstructuraFuncional', ns):
            lines.extend(get_text(child, ns))
            lines.append('')

    # Clean up: collapse multiple spaces, normalize newlines, strip blank lines moderately
    cleaned = []
    for line in lines:
        # Decode numeric entities already handled by XML parser
        # Normalize internal whitespace
        line = re.sub(r'[ \t]+', ' ', line)
        line = line.strip()
        cleaned.append(line)

    # Write with single blank lines between blocks
    with open('procedimiento_penal.txt', 'w', encoding='utf-8') as f:
        prev_blank = False
        for line in cleaned:
            if line == '':
                if not prev_blank:
                    f.write('\n')
                prev_blank = True
            else:
                f.write(line + '\n')
                prev_blank = False

    print('Saved procedimiento_penal.txt')

if __name__ == '__main__':
    main()
