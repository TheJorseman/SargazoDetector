# SargazoDetector

Detector de sargazo dada una imagen.


## Instalación
Use el gestor de paquetes [pip](https://pip.pypa.io/en/stable/) para instalar los requerimientos.

```bash
pip install -r requirements.txt
```
O use el gestor de ambientes [anaconda](https://www.anaconda.com/) para instalar los requerimientos.

```bash
conda env create --name SargazoDetector --file=sargazo.yml
```
Activar el enviroment

```bash
conda activate SargazoDetector
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

## Contribuir
Las solicitudes de extracción son bienvenidas. Para cambios importantes, abra un problema primero para discutir qué le gustaría cambiar.

Asegúrese de actualizar las pruebas según corresponda.

## License
[GNU General Public License v3.0](https://choosealicense.com/licenses/gpl-3.0/)