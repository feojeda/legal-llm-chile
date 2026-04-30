import re

input_path = r"E:\workspace\codigo_penal.txt"
output_path = r"E:\workspace\codigo_penal_limpio.txt"

with open(input_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Patrones a eliminar (encabezados y pies de página)
remove_patterns = [
    r"^Codigo PENAL, JUSTICIA \(1874\)\s*$",
    r"^Biblioteca del Congreso Nacional de Chile.*$",
    r"^página \d+ de \d+\s*$",
    r"^Código PENAL\s*$",
    r"^CÓDIGO PENAL\s*$",
    r"^MINISTERIO DE JUSTICIA\s*$",
    r"^Fecha Publicación:.*$",
    r"^Tipo Versión:.*$",
    r"^Inicio Vigencia:.*$",
    r"^Fin Vigencia:.*$",
    r"^Url Corta:.*$",
    r"^Santiago, .*$",  # fechas de promulgación repetidas
    r"^Núm\. \d+\.-.*$",  # anotaciones como Núm. 2561
    r"^Anótese\.\s*$",
    r"^ERRÁZURIZ\.\s*$",
    r"^JOSE MARIA BARCELÓ\s*$",
    r"^Certificamos que la presente edición.*$",
    r"^EL PRESIDENTE DE LA REPÚBLICA\.\s*$",
    r"^Por cuanto el Congreso Nacional ha aprobado.*$",
    r"^siguiente\s*$",
    r"^CARLOS RIESCO\..*$",
    r"^M\. E\. BALLESTEROS\..*$",
    r"^RAMON C\. BRISEÑO\..*$",
]

compiled_patterns = [re.compile(p, re.IGNORECASE) for p in remove_patterns]

cleaned_lines = []
for line in lines:
    stripped = line.rstrip()
    if any(p.match(stripped) for p in compiled_patterns):
        continue
    cleaned_lines.append(stripped)

# Eliminar líneas vacías múltiples consecutivas (más de 2)
final_lines = []
empty_count = 0
for line in cleaned_lines:
    if line == "":
        empty_count += 1
        if empty_count <= 2:
            final_lines.append(line)
    else:
        empty_count = 0
        final_lines.append(line)

# Unir líneas que parecen ser continuación de párrafos (no inician con mayúscula ni son artículos)
# Pero para Causal LM, es mejor dejarlo como texto continuo con saltos de línea lógicos
final_text = "\n".join(final_lines)

with open(output_path, "w", encoding="utf-8") as f:
    f.write(final_text)

print(f"Archivo limpio guardado en: {output_path}")
print(f"Líneas originales: {len(lines)}")
print(f"Líneas limpias: {len(final_lines)}")
print(f"Palabras aproximadas: {len(final_text.split())}")
