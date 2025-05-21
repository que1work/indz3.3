Ось повний конспект у форматі Markdown (.md), створений на основі блокнота Kaggle "Imbalanced Classification: Advanced Algorithms" від Ashraf Khan:

---

# Алгоритми незбалансованої класифікації: Розширені методи

## Посилання на оригінальний блокнот

[Imbalanced Classification: Advanced Algorithms – Kaggle](https://www.kaggle.com/code/ashrafkhan94/imbalanced-classification-advanced-algorithms)

---

## 1. Вступ

Незбалансовані дані є поширеною проблемою в задачах класифікації, коли кількість прикладів одного класу значно перевищує кількість прикладів іншого. Це може призвести до того, що моделі будуть ігнорувати менш представлений клас, що особливо критично в задачах, де важливо виявляти рідкісні події, такі як шахрайство чи хвороби.

---

## 2. Бібліотеки та імпорт

У блокноті використовуються наступні бібліотеки:

* `pandas`, `numpy` – для обробки даних
* `matplotlib`, `seaborn` – для візуалізації
* `scikit-learn` – для побудови моделей та метрик
* `imblearn` – для методів балансування даних

---

## 3. Завантаження та попередня обробка даних

Використовується датасет "Oil Spill", який є класичним прикладом незбалансованих даних.([Kaggle][1])

```python
# Завантаження даних
df = pd.read_csv('oil-spill.csv')
```



**Примітка:** Датасет має значну диспропорцію між класами.

---

## 4. Аналіз дисбалансу класів

```python
# Перевірка розподілу класів
df['class'].value_counts()
```



**Висновок:** Клас "1" (oil spill) значно менш представлений, ніж клас "0" (не oil spill).

---

## 5. Розділення на навчальну та тестову вибірки

```python
from sklearn.model_selection import train_test_split

X = df.drop('class', axis=1)
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
```



---

## 6. Побудова базової моделі

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
```



**Висновок:** Базова модель має низьку точність для менш представленого класу.

---

## 7. Методи балансування даних

### 7.1. Oversampling (перевибірка меншого класу)

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```



**Примітка:** SMOTE створює синтетичні приклади меншого класу для балансування.([cai.sk][2])

### 7.2. Undersampling (зменшення більшого класу)

```python
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
```



**Примітка:** RandomUnderSampler випадково видаляє приклади більшого класу.

### 7.3. Комбіновані методи

```python
from imblearn.combine import SMOTEENN

smote_enn = SMOTEENN(random_state=42)
X_resampled, y_resampled = smote_enn.fit_resample(X_train, y_train)
```



**Примітка:** SMOTEENN поєднує пере- та недовибірку для досягнення кращого балансу.([journals.ekb.eg][3])

---

## 8. Побудова моделей після балансування

```python
model = RandomForestClassifier(random_state=42)
model.fit(X_resampled, y_resampled)
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
```



**Висновок:** Моделі після застосування методів балансування показують покращену точність для менш представленого класу.

---

## 9. Візуалізація результатів

```python
from sklearn.metrics import plot_confusion_matrix

plot_confusion_matrix(model, X_test, y_test, cmap=plt.cm.Blues)
plt.title('Матриця неточностей')
plt.show()
```



**Примітка:** Вставити скріншот матриці неточностей тут.

---

## 10. Тлумачення результатів

* **Базова модель**: Низька точність для менш представленого класу через дисбаланс.
* **Після SMOTE**: Покращення точності, але можливе перенавчання.
* **Після RandomUnderSampler**: Зменшення обсягу даних може призвести до втрати інформації.
* **Після SMOTEENN**: Найкращий баланс між точністю та обсягом даних.

---

## 11. Висновки

Методи балансування даних є критично важливими для покращення продуктивності моделей на незбалансованих наборах даних. Комбіновані методи, такі як SMOTEENN, часто забезпечують кращі результати, ніж окремі методи пере- або недовибірки.([cai.sk][2])

---

## 12. Глосарій термінів

* **SMOTE (Synthetic Minority Over-sampling Technique)**: Метод створення синтетичних прикладів меншого класу для балансування даних.
* **RandomUnderSampler**: Метод випадкового видалення прикладів більшого класу для балансування даних.
* **SMOTEENN**: Комбінація SMOTE та Edited Nearest Neighbors для покращення якості балансування.
* **Матриця неточностей (Confusion Matrix)**: Таблиця, що показує кількість правильних і неправильних передбачень моделі для кожного класу.
* **Точність (Precision)**: Частка правильних передбачень позитивного класу серед усіх передбачень позитивного класу.
* **Повнота (Recall)**: Частка правильних передбачень позитивного класу серед усіх реальних прикладів позитивного класу.
* **F1-міра**: Гармонічне середнє між точністю та повнотою.([journals.ekb.eg][3])

---

Цей конспект надає огляд методів обробки незбалансованих даних та їх вплив на продуктивність моделей машинного навчання.

---

[1]: https://www.kaggle.com/code/ashrafkhan94/oil-spill-imbalanced-classification?utm_source=chatgpt.com "Oil Spill | Imbalanced Classification - Kaggle"
[2]: https://www.cai.sk/ojs/index.php/cai/article/download/2022_4_981/1180/15796?utm_source=chatgpt.com "[PDF] NEW HYBRID DATA PREPROCESSING TECHNIQUE FOR HIGHLY ..."
[3]: https://journals.ekb.eg/article_414893_9e92b6e04aa25efa9bcbeef5275ebfc0.pdf?utm_source=chatgpt.com "[PDF] Enhancing Fraud Detection in Imbalanced Datasets"
