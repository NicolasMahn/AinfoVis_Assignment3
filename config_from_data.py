import os
import yaml
import util
import json

def main():
    directory = "data"

    # List all entries in the directory
    entries = os.listdir(directory)
    # Filter out only directories
    folders = [entry for entry in entries if os.path.isdir(os.path.join(directory, entry)) and entry != 'precalculated_data']

    config = util.load_config()
    data_topics = config['data_topics']

    if data_topics is None:
        data_topics = {}

    for folder in folders:
        with open(f"{directory}/{folder}/metadata.json", 'r', encoding='utf-8') as file:
            metadata = json.load(file)

        if len(metadata) > 0 and all(folder not in data_topics[topic]['topic_dir'] for topic in data_topics):
            # print(os.listdir(os.path.join(directory, folder, 'documents')))
            # if os.path.exists(os.path.join(directory, folder, 'documents')) and os.listdir(os.path.join(directory, folder, 'documents')):
            # #TODO Fix this

                if len(metadata) > 0:
                    data_topics[folder] = {
                        'topic_name': metadata[0]["source_name"],
                        'topic_dir': f"{directory}/{folder}",
                    }

    config['data_topics'] = data_topics

    print(config)
    with open('config.yaml', 'w', encoding='utf-8') as file:
        yaml.dump(config, file)


if __name__ == "__main__":
    main()