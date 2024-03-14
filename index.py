import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from mpmath import mp
from factor_analyzer import FactorAnalyzer
from scipy.linalg import eigh
import re
from unidecode import unidecode
from flask import Flask, render_template, request, jsonify
import os

def crear_app():
    app = Flask(__name__)

    @app.route("/")
    def index():    
        # Luego renderiza la plantilla HTML
        return render_template('index.html')

    @app.route('/procesar_texto', methods=['POST'])
    def procesar_texto():
        texto_escrito = request.form.get('numb')

        # Convertir caracteres con acentos a su forma sin acentos
        texto_sin_acentos = unidecode(texto_escrito)

        # Eliminar caracteres especiales
        texto_sin_especiales = re.sub(r'[^a-zA-Z0-9\s]', '', texto_sin_acentos)

        # Dividir el texto en palabras
        palabras = texto_sin_especiales.split()

        # Calcular la cantidad total de palabras y la longitud de cada parte
        total_palabras = len(palabras)
        print("Total de palabras:", total_palabras)
        longitud_parte = total_palabras // 3
        print(longitud_parte)
        resto = total_palabras % 3
        print(resto)

        # Distribuir las palabras de manera equitativa
        #parte1 = palabras[:longitud_parte]
        #parte2 = palabras[longitud_parte:2*longitud_parte]
        #parte3 = palabras[2*longitud_parte:3*longitud_parte]

        # Ajustar partes si hay resto
        #if resto == 1:
        #    parte3.append(palabras[-1])
        #elif resto == 2:
        #    parte2.append(palabras[-2])
        #    parte3.append(palabras[-1])

        # Distribuir las palabras de manera equitativa
        parte1 = palabras[:longitud_parte]
        parte2 = palabras[longitud_parte:2*longitud_parte]
        parte3 = palabras[2*longitud_parte:3*longitud_parte]

        print(len(parte1))
        print(len(parte2))
        print(len(parte3))

        # Ajustar partes si hay resto
        if resto == 1:
            parte3.append(palabras[-1])
        elif resto == 2:
            parte2.append(palabras[-2])
            parte3.append(palabras[-1])
            

        def contar_caracteres(palabras):
            conteo_caracteres = {i: 0 for i in range(1, 13)}

            for palabra in palabras:
                longitud = len(palabra)
                if longitud >= 12:
                    conteo_caracteres[12] += 1
                else:
                    conteo_caracteres[longitud] += 1

            return conteo_caracteres

        # Contar y clasificar palabras por longitud en cada parte
        longitudes_parte1 = contar_caracteres(parte1)
        longitudes_parte2 = contar_caracteres(parte2)
        longitudes_parte3 = contar_caracteres(parte3)
        print(longitudes_parte1)
        print(longitudes_parte2)
        print(longitudes_parte3)

        # Sumar el contenido del arreglo longitudes_parte1
        suma_longitudes_parte1 = sum(longitudes_parte1.values())
        suma_longitudes_parte2 = sum(longitudes_parte2.values())
        suma_longitudes_parte3 = sum(longitudes_parte3.values())

        # Imprimir la suma o hacer lo que desees con ella
        print("Suma de longitudes en parte1:", suma_longitudes_parte1)
        print("Suma de longitudes en parte2:", suma_longitudes_parte2)
        print("Suma de longitudes en parte3:", suma_longitudes_parte3)

        # Guardar resultados en variables
        resultados = {
            'Parte1': list(longitudes_parte1.values()),
            'Parte2': list(longitudes_parte2.values()),
            'Parte3': list(longitudes_parte3.values())
        }
        #'Parte1': [0, 6, 5, 2, 5, 2, 2, 2, 3, 0, 0, 0],
        #'Parte2': [2, 11, 2, 4, 0, 2, 2, 0, 2, 1, 1, 0],
        #'Parte3': [2, 6, 4, 1, 6, 2, 3, 0, 4, 0, 0, 0],
        

        # Crea un DataFrame de pandas
        df = pd.DataFrame(resultados)

        # Calcula la matriz de correlaciones
        correlation_matrix = df.corr()

        # Imprime la matriz de correlaciones
        print("Matriz de correlaciones:")
        print(correlation_matrix)

        #otro
        # Calcula la matriz de correlaciones
        correlation_matrix = df.corr()

        # Calcula los autovalores y autovectores de la matriz de correlaciones
        eigenvalues, eigenvectors = np.linalg.eig(correlation_matrix)

        # Ordena los autovalores y autovectores en orden descendente
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Calcula las cargas factoriales (componentes principales)
        loadings = np.sqrt(eigenvalues) * eigenvectors
        
        # Extrae las tres primeras columnas
        first_three_loadings = loadings[:, :1]

        # Verifica si hay al menos un valor negativo en el resultado
        if np.any(first_three_loadings < 0):
        # Invierte el signo de todos los valores
            first_three_loadings = -first_three_loadings

        # Formatea y imprime las cargas factoriales de los tres primeros componentes principales
        formatted_loadings = np.round(first_three_loadings, 8)
        formatted_loadings_str = np.array2string(formatted_loadings, separator=', ')

        #calculo final
        #Componentes Principales
        # Multiplica cada carga factorial por sí misma (elemento por elemento)
        squared_loadings = first_three_loadings * first_three_loadings

        # Formatea y imprime las cargas factoriales al cuadrado de los tres primeros componentes principales
        formatted_squared_loadings = np.round(squared_loadings, 8)
        formatted_squared_loadings_str = np.array2string(formatted_squared_loadings, separator=', ')

        # Calcula 1 - (valor al cuadrado)
        one_minus_squared_loadings = 1 - squared_loadings

        # Formatea y imprime los resultados
        formatted_one_minus_squared_loadings = np.round(one_minus_squared_loadings, 8)
        formatted_one_minus_squared_loadings_str = np.array2string(formatted_one_minus_squared_loadings, separator=', ')

        # Calcula la sumatoria de 1 - (cargas factoriales al cuadrado)
        sum_one_minus_squared_loadings = np.sum(one_minus_squared_loadings)

        # Calcula la sumatoria de las cargas factoriales
        sum_of_loadings = np.sum(first_three_loadings)

        # Multiplica la sumatoria por sí misma
        result = sum_of_loadings * sum_of_loadings

        ComponentesPrincipales = np.round(result/(result + sum_one_minus_squared_loadings), 4)
        print("este es el resultado Componentes Principales: ",  ComponentesPrincipales)

        #Datos necesarios para devolver
        #Componentes Principales
        print(ComponentesPrincipales)
        #Cargas Factoriales Componentes Principales
        print(formatted_loadings_str)
        cargasFactorialesCP = [float(match) for match in re.findall(r'-?\d+\.\d+', formatted_loadings_str)]
        print(cargasFactorialesCP)
        #Matriz de frecuencias
        parte1List = list(longitudes_parte1.values())
        parte2List = list(longitudes_parte2.values())
        parte3List = list(longitudes_parte3.values())

        return jsonify({'Componentes_Principales': ComponentesPrincipales, 'longitudes_parte1': parte1List, 'longitudes_parte2': parte2List, 'longitudes_parte3': parte3List, 'cargasFactorialesCP': cargasFactorialesCP})
    
    return app

if __name__ == '__main__':
    app = crear_app()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))