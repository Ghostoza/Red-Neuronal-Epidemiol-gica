import numpy as np
import tensorflow as tf
import pandas as pd

class EpidemiologicalRiskModel:
    def __init__(self): #self, cuando se lo ponemos a las variables le decimos al programa que pertenece a la clase entera y para mandarlo llamar siempre tenemos que usar el self, no se borra como en cambio de la variables locales
        #Como vamos a definir variables protegidas en la clase no es necesario darlas como argumnetos en el constructor, ya que no estamos obligando a darle esos datos al instante cuando construyamos el objeto, nosotros vamos a fabricar esos datos internamente usando numpy
        
        #Variables encapsuladas
        self._datos_crudos = None
        self._datos_pacientish = None
        self._modelish = None #El none es para apartar un espacio en la memoria, en este momento no tiene nada la variable protegida, pero mas adelante la voy a llenar con una matriz
    
    #Metodo para crear esos datos sinteticos   
    def preparar_datish(self):
        #Generamos 4 variables locales usando numpy para hacer nuestros registros
        
        age = np.random.randint(15,61, size=(1000,1))
        anual_couples = np.random.randint(1,11, size=(1000,1))
        use_of_protection = np.random.randint(0,101, size=(1000,1))
        antecedentes = np.random.randint(0,2, size=(1000,1))
        
        #Llamamos a nuestra funcion privada y la guardamos en otra protegida, pero solo con el _ guin bajo se vuelve protegida, ya que con ese guion es como una regla no escrita para que cuando alguien quiera modificarla no lo haga, sigue siendo accesible pero logicamente esta protegida por ese acuerdo
        
        self._respuestas_pacientish_y = self._generar_etiquetish(anual_couples, use_of_protection, antecedentes)
        
        #Empaquetamos nuestras columnas de manera local
        column_pacientish = np.column_stack((age,anual_couples,use_of_protection,antecedentes))
        
        # Guardamos la versión legible para Pandas
        self._datos_crudos = column_pacientish
        
        #Sacamos el maximo y minimo de los valores 
        value_min = np.min(column_pacientish, axis=0) #Para que normalice esos datos columna por columa y no de toda la matriz
        value_max = np.max(column_pacientish, axis=0)
        
        
        #******
        #Aplicamos la formula de normalizacion matematica para normalizar esos datos y no tener datos muy variados y todos tengan una misma metrica de 0 a 1
        datos_normalisidish = (column_pacientish-value_min)/(value_max-value_min)
        
        #Guardamos los valores en una matriz limpia en una variable protegida y por que protegida si no hay diferencia ya que lo dijiste que tiene que ver o no
        self._datos_pacientish = datos_normalisidish
    
    #Generamos un metodo privado para las etiquetas en este caso las respuestas, ya que si le damos los datos al azar, no encontraremos ningun patron, basicamente una algoritmo guiado ya que sabemos las respuestas solo lo entrenamos para que al momento de que analice los datos pueda predecirlos
    #Como las funciones entran como parametros se comportan como variable locales, por lo tanto no llevan self
    def _generar_etiquetish(self, parejas_anuales, uso_proteccion, antecedentes):
        
        #******
        #Un solo paciente solo esta en riesgo siiiii pasa eso, e inventamos esas metricas o logica para que el modelo tensorflow tenga algo matematico que descubrir
        respuestish = np.where(((parejas_anuales >= 3 ) & (uso_proteccion < 50 )) | ((antecedentes == 1) & (uso_proteccion < 80 )), 1,0)
        
        #Retornamos si tiene riego o no
        return respuestish
    
    
    def construir_entrenar_modelish(self):
        #Para armar este modelo nos vamos a enfocar en 3 pasos esenciales
        #1.-La estructura del modelo
        #2.-Las reglas
        #3.-Y las acciones que tomara
        
        #Primero a nuestra variable protegida vamos a asignarle una secuencia de capas de tenserflow con keras.sequential()
        
        self._modelish = tf.keras.Sequential([
            #Vamos a asignarle las capas en este caso seran 3 no se por que
            tf.keras.layers.Dense(16, activation='relu'), #lo unico pues solo las neuronas
            tf.keras.layers.Dense(8, activation='relu'), #
            tf.keras.layers.Dense(1, activation='sigmoid')#Aqui si se por que solo una ya que solo queremos que nos regreso solo una respuesta por que una sola neurona hace eso
        ])
        
        #Terminos
        #Keras.-Motor matematico escrito en C++, que es dificil de leer. Keras es la interfaz amigable escrita en python
        #Layers.-Capas los bloques de construccion
        #Dense.-Significa que cada neurona de esta capa esta conectada con todas las neuronas de la capa anterior y todas de la siguiente. Basicamente una multiplicacion de matrices
        #Sequencial.-Le dice a tenserflow que la informacion que va a viajar como en una linea de ensamble de fabrica de una sola via. Los datos entran a la capa 1, luego capa2, luego capa 3. (Sin saltos raros ni ciclos)
        #ReLU.- Es la funcion de activacion para las capas ocultas (su regla f(x) = max(0,x)), para evitar valores como lo son -5 lo apaga y convierte ese valor en 0 y si fuera 8 lo deja pasar intacto para que no se confunda con calculos con calculos negativos que son practicamente basura. Ya que es un estandar mundial
        #Sigmoid(Sigmoide).- Funcion de activacion obligatoria para la ultima capa. (S(x)=1/1+e^-x) La cual sirve si la red avienta un numero loquisimo como -500 o 2000, la funcion sigmoide lo normaliza o aplasta en este caso para que se un numero deciamal entre 0 y 1 ya que queremos predecir una probabilidad de 0 a 100 porciento
        
        #Embudamiento (Funneling) tecnica para que darle un espacio matematico para buscar las combinaciones complejas (edad con proteccion por ejemplo), en la siguiente reducimos de 16 a 8 para resumir y quedarse con los patrones mas importantes y finalmente colpasamos todo a una sola neurona de salida
        #Compile.- preparamos el terreno antes de empezar con dos parametros, Binary Cross Entropy y Adam (Optimizados)
        #Binary Cross Entropy.- Mide que tan mezclados estan los datos, mide la distancia matematica entre lo que la red adivino y la respuesta real, mientras mas cerca de 0 menos el algoritmo esta confundido
        #Adam (Optimizador).- Es el algoritmo de Calculo Multivariable, cuando la red se equivoca, Adam calcula derivadas parciales para saber en que direccion ajustar los pesos (numeros internos) de las neuronas para que en el siguiente Epoch (epoca) el error sea menor. (Boosting reference)
        #evaluate().-Sirve para medir que tan bien aprendio despues de haber entrenado. Funciona igual que fit, pero sin modificar nada, la red neuronal usa los datos de entrada (x) hace calculos y calculo matematicamente la distancia entre lo que adivino y la realidad (usando la formula de binary cross entropy y devuelve un promedio de los errores
        #predict().-Toma los datos crudos del paciente (x) y los pasa por el embudo de las neuronas usando los pesos matematicos exactos que aprendio durante el entrenamiento y al final, la capa del sigmoide aplasta ese resultado para darte un numero entre 0 o 1, basicamente un porcentaje y lleva solo x ya que hace la prediccion en base a lo que aprendio en los patrones pasados
        
        #Luego seguimos con las reglas para decirle a nuestro modelo como se calificara cuando se equivoque o que metricas usara
        self._modelish.compile(optimizer='adam', loss='binary_crossentropy')#Nose para que sirve exactamente o que hace, tanto compile como el optimizar como ese adam, e imagino que ese de crossentropy es como la impuridad de giny entre mas cerca este de cero mejor ya que da resultados entre 0 y 1 y los porcentajes que de es que tanto se equivoco
        
        #Luego lo entrenamos, usando los datos que normalizamos para que nuestra red neuronal se ponga a trabajar
        self._modelish.fit(x=self._datos_pacientish, y=self._respuestas_pacientish_y, epochs=15, validation_split = 0.2)#por parametro le damos, los datos de los pacientes X y las labels con el patron que creamos sea 0 o 1 si esta o no en riesgo, luego epochs, epocas para ver cuantas veces lo hace para que aprenda bien el patron y finalmente separar el 20 porciento de los pacientes como test, como lo habiamos hecho en materias de machine learning separar la base de datos 80/20 trainin y test para los modelos, como bagging, random_forest y boosting
    
    #Funcion para evaluar el modelo que tan efectivo es    
    def evaluar_modelish(self):
        
        nivel_error = self._modelish.evaluate(x=self._datos_pacientish,y=self._respuestas_pacientish_y)
        print(f"El nivel de error es: {nivel_error}")
        
        pacientish_prove = self._datos_pacientish[:5] #Tomamos los 5 pacientes limpios, para que despues nuestro algoritmo adivine el riesgo de esos 5 
        
        prediccionsish = self._modelish.predict(x=pacientish_prove)
        
        print("Reporte de los 5 pacientes")
        # Recorremos los 5 resultados
        for i in range(5):
            # Extraemos el numero de la matriz y lo hacemos porcentaje
            probabilidad = prediccionsish[i][0] * 100 
            
            # Ponemos un semáforo visual con un if en una sola linea
            alerta = "ALTO RIESGO" if probabilidad >= 50 else "BAJO RIESGO"
            
            # Imprimimos con f-strings limitando a 2 decimales (.2f)
            print(f" Paciente {i+1} | Probabilidad: {probabilidad:.2f}% -> {alerta}")
            
    #metodo opcional para mostrar todos los pacientes
    def mostrar_pacientes_pandas(self, cantidad=10):
       
        print("  VISUALIZADOR DE EXPEDIENTES (PANDAS) ")
        
    
        pd.set_option('display.max_columns', None) # Muestra todas las columnas
        pd.set_option('display.width', 1000)       # Amplía el ancho de la pantalla virtual
        
        nombres_columnas = ['Edad', 'Parejas', 'Proteccion', 'Antecedentes']
        tabla = pd.DataFrame(self._datos_crudos[:cantidad], columns=nombres_columnas)
        tabla['Riesgo Real'] = self._respuestas_pacientish_y[:cantidad]
        
        print(tabla)
        
    
    def predecir_paciente_manual(self):
        while(True):
            print(" BÚSQUEDA MANUAL DE PACIENTE ")
            
            # Usamos un try/except para que no se rompa el programa si el usuario escribe letras en vez de numeros
            try:
                # Le pedimos al usuario que escriba un ID (como tenemos 1000 pacientes, va del 0 al 999)
                id_paciente = int(input("Ingrese el ID del paciente a evaluar (0 - 999): "))
                
                # Verificamos que el numero exista en nuestra base de datos
                if 0 <= id_paciente < 1000:
                    
                    # Agarramos los datos de ESE paciente en especifico y usamos reshape(1, -1) 
                    # para que TensorFlow lo acepte como un "lote de 1 paciente"
                    datos_del_paciente = self._datos_pacientish[id_paciente].reshape(1, -1)
                    
                    # Predecimos usando verbose=0 para que no imprima barritas de carga feas en la consola
                    prediccion = self._modelish.predict(datos_del_paciente, verbose=0)
                    probabilidad = prediccion[0][0] * 100
                    
                    # Nuestro semáforo visual
                    alerta = "ALTO RIESGO" if probabilidad >= 50 else " BAJO RIESGO"
                    
                    print(f" RESULTADO EXPEDIENTE #{id_paciente}")
                    print(f"Probabilidad Calculada por IA: {probabilidad:.2f}% -> {alerta}\n")
                    
                else:
                    print(" ID fuera de rango. Intente con un número del 0 al 999.")
                    
            except ValueError:
                print(" Error: Ingresó un texto. Debe ingresar un número entero.")
            respuesta =input("Quieres ver otro paciente? y/n")
            if respuesta.lower() == 'n':
                break

if __name__ == "__main__":
    
    modelish_ETS =EpidemiologicalRiskModel()
    
    #Llamar al método para crear y normalizar los datos
    modelish_ETS.preparar_datish()
    
    #Llamar al método para construir la red y entrenarla
    modelish_ETS.construir_entrenar_modelish()
    
     #Llamar al método final para evaluar y predecir
    modelish_ETS.evaluar_modelish()
    
    option = 0
    while(option!=3):
        print("Menu")
        print("What do you need?")
        print("1.-Show specific number of patients")
        print("2.-Search an specific patient and predict it")
        print("3.-Exit to the program")
        option = int(input("Select your number: "))
        
        if option == 1:
            # Mostramos n pacientes que quiera el usuario pacientes en una tabla bonita
            n_patients = int(input("How many patients do you want?"))
            
            modelish_ETS.mostrar_pacientes_pandas(cantidad=n_patients)
        elif option == 2:
            # Hacemos que el programa interactúe con el usuario
            # (Puedes meter esto en un ciclo 'while True' si quieres que pregunte muchas veces, 
            # pero con llamarlo una vez está perfecto para la demostración).
            modelish_ETS.predecir_paciente_manual()
        elif option == 3:
            break
        else: print("Your number is doesn't exist, type again")