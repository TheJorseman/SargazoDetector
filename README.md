# Sargassum Segmentation and Optical Flow 

Detector de sargazo dada una imagen.


## Instalación
Use el gestor de paquetes [pip](https://pip.pypa.io/en/stable/) para instalar los requerimientos.

```bash
pip install -r requirements.txt
```
O use el gestor de ambientes [anaconda](https://www.anaconda.com/) para instalar los requerimientos.

```bash
conda env create --name sargazo--file=sargazo.yml
```
Activar el enviroment

```bash
conda activate sargazo
```

## Usage

Con hardcoding para introducir manualmente las carpetas.

```bash
python segmentation.py
```

Usando el programa con argumentos.

```bash
python segmentation_args.py --mask_folder=Masked --root_folder=full_data_1 --output_folder=full_data_1_gnb --mask_path=mask.png --use_gnb=True
```

Para segmentar una imagen se utiliza el siguiente comando:

```
usage: segmentation_predict.py [-h] [--img_path IMG_PATH] [--output OUTPUT]

optional arguments:
-h, --help           show this help message and exit
--img_path IMG_PATH  Image Path
--output OUTPUT      Output Path
```

Ejemplo:

```bash
python segmentation_predict.py --img_path sargazo-cortesia.jpg --output=sargazo-cortesia-mask.png
```

Calculo del Flujo Óptico

```bash
python optical_flow_predict.py 
```

Calculo de la segmentación y del flujo óptico

```bash
python main.py --mask_folder=Masked --root_folder=full_data_1 --output_folder=full_data_1_gnb --mask_path=mask.png --use_gnb=True
```


## Contribuir
Las solicitudes de extracción son bienvenidas. Para cambios importantes, abra un problema primero para discutir qué le gustaría cambiar.

Asegúrese de actualizar las pruebas según corresponda.

## License
[GNU General Public License v3.0](https://choosealicense.com/licenses/gpl-3.0/)