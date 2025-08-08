import warnings # Suprimir advertencias de TensorFlow
import os # Operaciones del sistema operativo

warnings.filterwarnings("ignore") # Filtrar advertencias molestas del sistema
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Establecer nivel de logging mínimo

import pathlib # Manejo moderno de rutas de archivos y directorios
import librosa # Librosa para validación y carga de audio
import soundfile # Para escribir archivos de audio
from spleeter.separator import Separator # Spleeter para separación de audio

# FUNCIONES
# Guarda los archivos de audio separados con nomenclatura apropiada
def guardar_audios_separados(dir_audio, audio_vocal, audio_instrumental, carpeta_destino, directorio_salida):
    # Crear directorio de salida si no existe
    pathlib.Path(directorio_salida).mkdir(exist_ok = True)
        
    cont_audios = 0  # Contador de archivos guardados
    
    try:
        # Guardar audio vocal
        ruta_vocal = carpeta_destino / f"{dir_audio.stem} (vocal).wav"
        soundfile.write(str(ruta_vocal), audio_vocal, 44100)
        
        cont_audios += 1
        
        # Guardar audio instrumental
        ruta_instrumental = carpeta_destino / f"{dir_audio.stem} (music).wav"
        soundfile.write(str(ruta_instrumental), audio_instrumental, 44100)
        
        cont_audios += 1  
    except Exception as e:
        return 0 # Retornar cero si hubo error al guardar
    
    return cont_audios # Retornar cantidad de audios guardados exitosamente

# Separa audio en vocal e instrumental usando Spleeter
def separar_audio(ruta_audio, spleeter_modelo):
    try:
        # Cargar audio usando librosa con formato específico
        audio_data, sample_rate = librosa.load(str(ruta_audio), sr = 44100, mono = False)
        
        # Asegurar formato estéreo correcto para Spleeter
        if len(audio_data.shape) == 1:
            # Mono a estéreo
            audio_data = audio_data.reshape(-1, 1)
            audio_data = audio_data.repeat(2, axis = 1)
        elif audio_data.shape[0] == 2:
            # Si está en formato (2, samples), transponerlo a (samples, 2)
            audio_data = audio_data.T
        
        # Separar audio usando Spleeter
        waveform_val = spleeter_modelo.separate(audio_data)
        
        # Extraer vocal e instrumental
        audio_vocal = waveform_val['vocals']
        audio_instrumental = waveform_val['accompaniment']
        
        return audio_vocal, audio_instrumental
    except Exception as e:
        return None, None # Retornar valores nulos si hubo error en la separación

# Procesamiento de archivos de audio y guardado de resultados
def procesar_directorio_audios(directorio_entrada, extensiones_lista, directorio_salida, spleeter_modelo):
    # Contadores para estadísticas finales
    total_archivos_procesados = 0
    total_audios_separados = 0
    
    # Procesar recursivamente cada archivo de audio en el directorio
    for extension_val in extensiones_lista:
        for archivo_iter in pathlib.Path(directorio_entrada).rglob(f'*.{extension_val}'):
            if archivo_iter.is_file():
                # Generar ruta relativa para mantener estructura de directorios
                carpeta_destino = pathlib.Path(directorio_salida) / archivo_iter.parent.relative_to(pathlib.Path(directorio_entrada))
                
                # Crear carpeta de destino si no existe
                carpeta_destino.mkdir(parents = True, exist_ok = True)
                
                # Separar audio en vocal e instrumental
                audio_vocal, audio_instrumental = separar_audio(archivo_iter, spleeter_modelo)
                
                # Verificar si la separación fue exitosa
                if audio_vocal is not None and audio_instrumental is not None:
                    # Guardar audios separados
                    audios_guardados = guardar_audios_separados(archivo_iter, audio_vocal, audio_instrumental, carpeta_destino, directorio_salida)
                    
                    # Actualizar variable
                    if audios_guardados > 0:
                        total_audios_separados += audios_guardados
                
                total_archivos_procesados += 1
    
    # Mostrar separador visual para inicio de resultados
    print("\n" + "-" * 36)

    # Mostrar mensaje si no se procesaron archivos
    if total_archivos_procesados == 0:
        print("No audio files found")
    else:
        if total_audios_separados == 0:
            print("No audio tracks generated")
        else:
            print(f"Processed audio files: {total_archivos_procesados}")
            print(f"Generated audio tracks: {total_audios_separados}")

    # Mostrar separador final
    print("-" * 36 + "\n")

# Inicializa el modelo Spleeter para separación de audio
def cargar_modelo_spleeter():
    try:
        # Cargar modelo 2stems-16kHz (vocal/accompaniment) de Spleeter
        separator_val = Separator('spleeter:2stems-16kHz', multiprocess = False)
        
        return separator_val
    except Exception as e:
        return None # Retornar valor nulo si no se pudo cargar el modelo

# PUNTO DE PARTIDA
# Lista de formatos de audio soportados por Spleeter y librosa
extensiones_lista = ['mp3', 'wav', 'flac', 'm4a', 'aac', 'ogg', 'wma']

try:
    # Cargar modelo Spleeter (solo una vez)
    spleeter_modelo = cargar_modelo_spleeter()
    
    # Verificar si el modelo se cargó correctamente
    if spleeter_modelo is None:
        raise Exception() # Forzar excepción si el modelo no se cargó
        
    # Bucle principal del programa
    while True:
        # Solicitar directorio de entrada
        while True:
            directorio_entrada = input("Enter directory: ").strip('"\'')

            print() # Salto de línea
            
            # Verificar que el directorio exista
            if not pathlib.Path(directorio_entrada).exists():
                print("Wrong directory\n")
            else:
                break # Salir del bucle si el directorio es válido
        
        # Generar nombre para directorio de salida
        directorio_salida = f"{pathlib.Path(directorio_entrada).name} (output)"
        
        # Ejecutar separación de audio y guardado con procesamiento durante ejecución
        procesar_directorio_audios(directorio_entrada, extensiones_lista, directorio_salida, spleeter_modelo)  
except Exception as e:
    print("Spleeter model not found")
    
    # Esperar entrada del usuario antes de cerrar
    input()
