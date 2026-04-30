import fitz  # PyMuPDF
import os

pdf_path = r"C:\Users\feoje\Downloads\Codigo-PENAL_12-NOV-1874.pdf"
output_path = r"E:\workspace\codigo_penal.txt"

doc = fitz.open(pdf_path)
text_parts = []

for page in doc:
    text = page.get_text()
    text_parts.append(text)

full_text = "\n".join(text_parts)

# Limpieza básica: eliminar múltiples espacios y líneas vacías excesivas
lines = full_text.splitlines()
cleaned_lines = []
for line in lines:
    stripped = line.strip()
    if stripped:
        cleaned_lines.append(stripped)
    else:
        # Mantener una sola línea vacía como separador de párrafos
        if cleaned_lines and cleaned_lines[-1] != "":
            cleaned_lines.append("")

# Unir en texto continuo con líneas separadas
cleaned_text = "\n".join(cleaned_lines)

with open(output_path, "w", encoding="utf-8") as f:
    f.write(cleaned_text)

print(f"Texto extraído y guardado en: {output_path}")
print(f"Total de páginas: {len(doc)}")
print(f"Caracteres totales: {len(cleaned_text)}")
print(f"Palabras aproximadas: {len(cleaned_text.split())}")

doc.close()
