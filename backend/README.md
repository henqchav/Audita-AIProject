# Backend del Proyecto

Este es el backend del proyecto que forma parte de un monorepo. Aquí encontrarás las instrucciones para configurar el entorno de desarrollo y ejecutar la aplicación.

---

## **Requisitos Previos**

Antes de comenzar, asegúrate de tener lo siguiente instalado en tu sistema:

1. **Python** (versión: 3.10.12)
2. **Git**
3. pip 24.3.1

---

## **Configuración del Entorno**

### **1. Clonar el Repositorio**

Clona el monorepo en tu máquina local:
```bash
git clone <URL_DEL_REPOSITORIO>
cd IA/back
```


Para que las personas que trabajen contigo puedan configurar un entorno de Python igual al tuyo, es importante proporcionar instrucciones claras en el archivo README del backend. Aquí tienes un ejemplo de lo que deberías incluir:
README.md para el Backend

# Backend del Proyecto

Este es el backend del proyecto que forma parte de un monorepo. Aquí encontrarás las instrucciones para configurar el entorno de desarrollo y ejecutar la aplicación.

---

## **Requisitos Previos**

Antes de comenzar, asegúrate de tener lo siguiente instalado en tu sistema:

1. **Python** (versión: X.X.X)
2. **Git**
3. (Opcional) **Make** si deseas usar comandos simplificados.

---

## **Configuración del Entorno**

### **1. Clonar el Repositorio**

Clona el monorepo en tu máquina local:
```bash
git clone <URL_DEL_REPOSITORIO>
cd IA/back
```

### **2. Crear y Activar un Entorno Virtual**

Usaremos un entorno virtual para gestionar dependencias. Sigue estos pasos:
Linux/MacOS:

```bash
python3 -m venv venv
source venv/bin/activate
```
Windows:

```bash
python -m venv venv
venv\Scripts\activate
```
Nota: Deberías ver algo como (venv) al principio de tu terminal indicando que el entorno virtual está activo.

### **3. Instalar las Dependencias**

Con el entorno virtual activo, instala las dependencias necesarias:

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```