import cv2
import os
import argparse
import imageio.v2 as imageio

def create_gif(input_directory, output_gif_path, duration=0.1):
    image_files = sorted([f for f in os.listdir(input_directory) if f.endswith('.png')], key=lambda x: int(''.join(filter(str.isdigit, x))))

    images = []
    for image_file in image_files:
        image_path = os.path.join(input_directory, image_file)
        images.append(imageio.imread(image_path))

    imageio.mimsave(output_gif_path, images, duration=duration)

    print(f'O GIF foi salvo em: {output_gif_path}')


def images_to_video(input_directory, output_video_path):
    image_files = sorted([f for f in os.listdir(input_directory) if f.endswith('.png')], key=lambda x: int(''.join(filter(str.isdigit, x))))
    print(image_files)

    # Obtendo as dimensões da primeira imagem para configurar o vídeo
    first_image_path = os.path.join(input_directory, image_files[0])
    first_image = cv2.imread(first_image_path)
    height, width, layers = first_image.shape

    # Escolhendo um codec suportado (H.264 neste exemplo)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Inicializando o objeto VideoWriter
    video = cv2.VideoWriter(output_video_path, fourcc, 1, (width, height))

    # Iterando sobre cada imagem e gravando no vídeo
    for image_file in image_files:
        image_path = os.path.join(input_directory, image_file)
        frame = cv2.imread(image_path)
        video.write(frame)

    # Liberando o objeto VideoWriter
    video.release()

    print(f'O vídeo foi salvo em: {output_video_path}')

def main():
    parser = argparse.ArgumentParser(description='Converte uma sequência de imagens PNG em um vídeo.')
    parser.add_argument('--dir_data_in', required=True, help='Caminho para o diretório de entrada contendo imagens PNG')
    parser.add_argument('--file_data_out', required=True, help='Nome do arquivo de saída do vídeo (sem extensão)')
    parser.add_argument('--duration', dest='duration', type=float, default=0.1, help='Giff duration.')

    args = parser.parse_args()

    input_directory = args.dir_data_in
    output_video_path = args.file_data_out + '.gif'

    create_gif(input_directory, output_video_path, duration=args.duration)

    # images_to_video(input_directory, output_video_path)

if __name__ == "__main__":
    main()
