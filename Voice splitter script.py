import collections # defaultdict para agrupar archivos por carpeta
import pathlib # Manejo moderno de rutas de archivos y directorios
import os # Operaciones del sistema operativo
import shutil # Operaciones de archivos y directorios

# Suprimir advertencias de TensorFlow
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from spleeter.separator import Separator # Spleeter para separación de audio
import librosa # Librosa para validación y carga de audio
import soundfile as sf # Para escribir archivos de audio

# FUNCIONES
# Guarda los archivos de audio separados con nomenclatura apropiada
def guardar_audios_separados(dir_audio, audio_vocal, audio_instrumental, carpeta_destino, directorio_salida):
    # Crear directorio de salida si no existe
    pathlib.Path(directorio_salida).mkdir(exist_ok=True)
    
    # Extraer nombre base y extensión del archivo original
    nombre_archivo = dir_audio.stem
    extension_archivo = ".wav"  # Spleeter genera archivos WAV
    
    cont_audios = 0  # Contador de archivos guardados
    
    try:
        # Guardar audio vocal
        ruta_vocal = carpeta_destino / f"{nombre_archivo} (vocal){extension_archivo}"
        sf.write(str(ruta_vocal), audio_vocal, 44100)
        cont_audios += 1
        
        # Guardar audio instrumental
        ruta_instrumental = carpeta_destino / f"{nombre_archivo} (music){extension_archivo}"
        sf.write(str(ruta_instrumental), audio_instrumental, 44100)
        cont_audios += 1
        
    except Exception as e:
        return 0
    
    return cont_audios

# Separa audio en vocal e instrumental usando Spleeter
def separar_audio(ruta_audio, separador_modelo):
    try:
        # Cargar audio usando librosa
        audio_data, sample_rate = librosa.load(str(ruta_audio), sr=44100, mono=False)
        
        # Si el audio es mono, convertirlo a estéreo
        if len(audio_data.shape) == 1:
            audio_data = audio_data.reshape(1, -1)
            audio_data = audio_data.repeat(2, axis=0).T
        else:
            audio_data = audio_data.T
        
        # Separar audio usando Spleeter
        waveform = separador_modelo.separate(audio_data)
        
        # Extraer vocal e instrumental
        audio_vocal = waveform['vocals']
        audio_instrumental = waveform['accompaniment']
        
        return audio_vocal, audio_instrumental
        
    except Exception as e:
        return None, None

# Procesamiento de archivos de audio y guardado de resultados
def procesar_directorio_audios(directorio_entrada, lista_carpetas_archivos, directorio_salida, separador_modelo):
    # Contadores
    total_audios_separados = 0
    total_archivos_procesados = 0
    
    # Mostrar separador visual para inicio de resultados
    print("-" * 36)
    
    # Procesar cada carpeta y sus archivos de audio
    for iter_carpeta, lista_archivos in lista_carpetas_archivos.items():
        # Generar ruta relativa para mantener estructura de directorios
        carpeta_destino = pathlib.Path(directorio_salida) / pathlib.Path(iter_carpeta).relative_to(pathlib.Path(directorio_entrada))
        
        # Crear carpeta de destino si no existe
        carpeta_destino.mkdir(parents=True, exist_ok=True)
        
        # Procesar cada archivo de audio individual de la carpeta actual
        for dir_audio in lista_archivos:
            print(f"Processing: {dir_audio.name}")
            
            # Separar audio en vocal e instrumental
            audio_vocal, audio_instrumental = separar_audio(dir_audio, separador_modelo)
            
            # Guardar audios separados si se procesó correctamente
            if audio_vocal is not None and audio_instrumental is not None:
                audios_guardados = guardar_audios_separados(dir_audio, audio_vocal, audio_instrumental, carpeta_destino, directorio_salida)
                total_audios_separados += audios_guardados
                
                if audios_guardados > 0:
                    print(f"  ✓ Separated into vocal and music tracks")
                else:
                    print(f"  ✗ Error saving separated tracks")
            else:
                print(f"  ✗ Error processing audio")
            
            total_archivos_procesados += 1
    
    return total_archivos_procesados, total_audios_separados

# Organiza archivos de audio agrupándolos por carpeta (búsqueda recursiva)
def agrupar_archivos_carpetas(directorio_origen, extensiones_lista):
    # Diccionario para carpetas y archivos de la misma
    dicc_carpetas_archivos = collections.defaultdict(list)
    
    # Buscar archivos para cada extensión de audio soportada
    for extension_val in extensiones_lista:
        # Búsqueda recursiva en todas las subcarpetas usando patrón glob
        for archivo_iter in pathlib.Path(directorio_origen).rglob(f'*.{extension_val}'):
            # Verificar que sea un archivo válido y no un directorio
            if archivo_iter.is_file():
                carpeta_contenedora = archivo_iter.parent
                dicc_carpetas_archivos[carpeta_contenedora].append(archivo_iter)
    
    return dicc_carpetas_archivos

# Inicializa el modelo Spleeter para separación de audio
def cargar_modelo_spleeter():
    try:
        # Cargar modelo 2stems-16kHz (vocal/accompaniment) de Spleeter
        separator = Separator('spleeter:2stems-16kHz')
        return separator
    except Exception as e:
        return None

# PUNTO DE PARTIDA
# Lista de formatos de audio soportados por Spleeter y librosa
extensiones_lista = ['mp3', 'wav', 'flac', 'm4a', 'aac', 'ogg', 'wma']

try:
    # Cargar modelo Spleeter
    print("Loading Spleeter model...")
    separador_modelo = cargar_modelo_spleeter()
    
    if separador_modelo is None:
        raise Exception("Failed to load Spleeter model")
    
    print("Model loaded successfully!\n")
    
    # Bucle principal del programa
    while True:
        # Solicitar directorio de entrada
        while True:
            directorio_entrada = input("Enter directory: ").strip('"\'')
            
            # Verificar que el directorio exista
            if not pathlib.Path(directorio_entrada).exists():
                print("Wrong directory\n")
            else:
                break
        
        # Generar nombre para directorio de salida
        directorio_salida = f"{pathlib.Path(directorio_entrada).name} (output)"
        
        # Generar lista de directorios de audios en carpeta de entrada
        lista_carpetas_archivos = agrupar_archivos_carpetas(directorio_entrada, extensiones_lista)
        
        # Ejecutar separación de audio y guardado
        total_archivos, total_audios = procesar_directorio_audios(directorio_entrada, lista_carpetas_archivos, directorio_salida, separador_modelo)
        
        # Mostrar separador visual para resultados
        print("-" * 36)
        
        # Mostrar resultados de procesamiento
        if total_archivos > 0:
            print(f"Processed files: {total_archivos}")
            
            # Mostrar cantidad de audios separados sólo si se procesó alguno
            if total_audios > 0:
                print(f"Generated tracks: {total_audios}")
            else:
                print("No tracks generated")
        else:
            print("No audio files found")
        
        print("-" * 36 + "\n")

except Exception as e:
    print("Spleeter model not available or dependencies missing")
    print("Please install: pip install spleeter librosa soundfile")
    
    # Detener el programa
    input()
