# Regresie Logistica cu Regularizare

Acest proiect implementeaza un model de regresie logistica cu regularizare L2 (Ridge) folosind NumPy si Matplotlib. Modelul foloseste Gradient Descent pentru optimizarea parametrilor si aplica functia sigmoid pentru clasificare binara.

## Structura Proiectului

Fisierele principale:

1. **cost_function.py** - Calculeaza costul pentru regresia logistica cu regularizare.
2. **gradient.py** - Contine functiile pentru calculul gradientului si algoritmul de Gradient Descent.
3. **sigmoid.py** - Implementarea functiei sigmoid.
4. **main.py** - Scriptul principal care antreneaza modelul si face predictii.

## Instalare si Configurare

### Cerinte
- Python 3.x
- NumPy
- Matplotlib

### Instalare
```bash
pip install numpy matplotlib
```

## Utilizare

1. **Antrenarea Modelului:**
   - Se initializeaza parametrii w si b.
   - Se ruleaza Gradient Descent pentru a optimiza parametrii.
   - Se utilizeaza regularizarea L2 pentru a preveni overfitting-ul.

2. **Predictii:**
   - Se aplica modelul antrenat pe datele de test.
   - Se utilizeaza functia sigmoid pentru a decide clasa (0 sau 1).

### Exemplu de rulare
```bash
python main.py
```

## Rezultate
Modelul afiseaza costul pe parcursul antrenarii si clasifica corect exemplele de test.

