import json
from graphqlclient import GraphQLClient
import glob
import os
import pandas as pd

TILE_DIR = '/Users/darylwilding-mcbride/Downloads/190719_Hela_Ecoli_1to1_01-tiles/tile-33/pre-assigned'
BASE_TILE_URL = 'https://dwm-tims.s3-ap-southeast-2.amazonaws.com/190719_Hela_Ecoli_1to1_01-tiles/tile-33'
TILE_EDGE_LENGTH_PIXELS = 910

client = GraphQLClient('https://api.labelbox.com/graphql')
client.inject_token('Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjanhmb3NwcjVnNnZ5MDcyMXV0M3dwNTZmIiwib3JnYW5pemF0aW9uSWQiOiJjanhmb3NwcWxnNWgwMDg0OGhjbXozdmx5IiwiYXBpS2V5SWQiOiJjanpkOWpiajN2a3N6MDcwMXpuMTU2cW1mIiwiaWF0IjoxNTY1OTA4NTc3LCJleHAiOjIxOTcwNjA1Nzd9.aYPO1bmk5EyF7XQ1eI4-bgc_BYi6f4AFSnkWU36_U14')

def me():
    res_str = client.execute("""
    query GetUserInformation {
      user {
        id
        organization{
          id
        }
      }
    }
    """)

    res = json.loads(res_str)
    return res['data']['user']


def createDataset(name):
    res_str = client.execute("""
    mutation CreateDatasetFromAPI($name: String!) {
      createDataset(data:{
        name: $name
      }){
        id
      }
    }
    """, {'name': name})

    res = json.loads(res_str)
    return res['data']['createDataset']['id']


def createProject(name):
    res_str = client.execute("""
    mutation CreateProjectFromAPI($name: String!) {
      createProject(data:{
        name: $name
      }){
        id
      }
    }
    """, {'name': name})

    res = json.loads(res_str)
    return res['data']['createProject']['id']


def completeSetupOfProject(project_id, dataset_id, labeling_frontend_id):
    res_str = client.execute("""
    mutation CompleteSetupOfProject($projectId: ID!, $datasetId: ID!, $labelingFrontendId: ID!){
      updateProject(
        where:{
          id:$projectId
        },
        data:{
          setupComplete: "2018-11-29T20:46:59.521Z",
          datasets:{
            connect:{
              id:$datasetId
            }
          },
          labelingFrontend:{
            connect:{
              id:$labelingFrontendId
            }
          }
        }
      ){
        id
      }
    }
    """, {
        'projectId': project_id,
        'datasetId': dataset_id,
        'labelingFrontendId': labeling_frontend_id
    })

    res = json.loads(res_str)
    return res['data']['updateProject']['id']


def configure_interface_for_project(ontology, project_id, interface_id, organization_id):
    res_str = client.execute("""
      mutation ConfigureInterfaceFromAPI($projectId: ID!, $customizationOptions: String!, $labelingFrontendId: ID!, $organizationId: ID!) {
        createLabelingFrontendOptions(data:{
          customizationOptions: $customizationOptions,
          project:{
            connect:{
              id: $projectId
            }
          }
          labelingFrontend:{
            connect:{
              id:$labelingFrontendId
            }
          }
          organization:{
            connect:{
              id: $organizationId
            }
          }
        }){
          id
        }
      }
    """, {
        'projectId': project_id,
        'customizationOptions': json.dumps(ontology),
        'labelingFrontendId': interface_id,
        'organizationId': organization_id,
    })

    res = json.loads(res_str)
    return res['data']['createLabelingFrontendOptions']['id']


def get_image_labeling_interface_id():
    res_str = client.execute("""
      query GetImageLabelingInterfaceId {
        labelingFrontends(where:{
          iframeUrlPath:"https://image-segmentation-v4.labelbox.com"
        }){
          id
        }
      }
    """)

    res = json.loads(res_str)
    return res['data']['labelingFrontends'][0]['id']


def create_prediction_model(name, version):
    res_str = client.execute("""
      mutation CreatePredictionModelFromAPI($name: String!, $version: Int!) {
        createPredictionModel(data:{
          name: $name,
          version: $version
        }){
          id
        }
      }
    """, {
      'name': name,
      'version': version
    })

    res = json.loads(res_str)
    return res['data']['createPredictionModel']['id']

def attach_prediction_model_to_project(prediction_model_id, project_id):
    res_str = client.execute("""
      mutation AttachPredictionModel($predictionModelId: ID!, $projectId: ID!){
        updateProject(where:{
          id: $projectId
        }, data:{
          activePredictionModel:{
            connect:{
              id: $predictionModelId
            }
          }
        }){
          id
        }
      }
    """, {
      'predictionModelId': prediction_model_id,
      'projectId': project_id
    })

    res = json.loads(res_str)
    return res['data']['updateProject']['id']


def create_prediction(label, prediction_model_id, project_id, data_row_id):
    res_str = client.execute("""
      mutation CreatePredictionFromAPI($label: String!, $predictionModelId: ID!, $projectId: ID!, $dataRowId: ID!) {
        createPrediction(data:{
          label: $label,
          predictionModelId: $predictionModelId,
          projectId: $projectId,
          dataRowId: $dataRowId,
        }){
          id
        }
      }
    """, {
        'label': label,
        'predictionModelId': prediction_model_id,
        'projectId': project_id,
        'dataRowId': data_row_id
    })

    res = json.loads(res_str)
    return res['data']['createPrediction']['id']


def create_datarow(row_data, external_id,dataset_id):
    res_str = client.execute("""
      mutation CreateDataRowFromAPI(
        $rowData: String!,
        $externalId: String,
        $datasetId: ID!
      ) {
        createDataRow(data:{
          externalId: $externalId,
          rowData: $rowData,
          dataset:{
            connect:{
              id: $datasetId
            }
          }
        }){
          id
        }
      }
    """, {
        'rowData': row_data,
        'externalId': external_id,
        'datasetId': dataset_id
    })

    res = json.loads(res_str)
    return res['data']['createDataRow']['id']


file_labels = []
for file in glob.glob("{}/*.txt".format(TILE_DIR))[:20]:
    tile_name = os.path.basename(file).split('.')[0]
    labels_df = pd.read_csv(file, sep=' ', names=['class_idx','x','y','width','height'])
    classes_d = {}
    for group_name,group_df in labels_df.groupby(['class_idx'], as_index=False):
        charge_state = int(group_name)+1
        geometry_l = []  # list of instances of this class
        for group_idx in range(len(group_df)):
            instance_label_df = group_df.iloc[group_idx]
            # x,y coordinates is the centre of the rectangle
            # origin for YOLO is top-left
            # origin for Labelbox is top-left
            centre_x_pixels = int(instance_label_df.x * TILE_EDGE_LENGTH_PIXELS)
            centre_y_pixels = int(instance_label_df.y * TILE_EDGE_LENGTH_PIXELS)
            width_pixels = int(instance_label_df.width * TILE_EDGE_LENGTH_PIXELS)
            height_pixels = int(instance_label_df.height * TILE_EDGE_LENGTH_PIXELS)

            x_lower_pixels = centre_x_pixels - int(width_pixels / 2)
            y_lower_pixels = centre_y_pixels - int(height_pixels / 2)
            x_upper_pixels = centre_x_pixels + int(width_pixels / 2)
            y_upper_pixels = centre_y_pixels + int(height_pixels / 2)

            vertices = []
            vertices.append({"x":x_lower_pixels,"y":y_lower_pixels})
            vertices.append({"x":x_upper_pixels,"y":y_lower_pixels})
            vertices.append({"x":x_upper_pixels,"y":y_upper_pixels})
            vertices.append({"x":x_lower_pixels,"y":y_upper_pixels})

            geometry_d = {"geometry":vertices}
            geometry_l.append(geometry_d)
        # classes_d is a dictionary of the classes, each dictionary entry is a list of geometries for each class instance
        classes_d["charge-{}".format(charge_state)] = geometry_l
        # consolidate the labels for this file
        file_label_d = {}
        file_label_d["prediction_label"] = classes_d
        file_label_d["image_url"] = "{}/{}.png".format(BASE_TILE_URL, tile_name)
        file_label_d["external_id"] = tile_name
    file_labels.append(file_label_d)


if __name__ == "__main__":
  user_info = me()
  org_id = user_info['organization']['id']
  project_id = createProject('Peptide feature labelling')
  print('Created project: %s' % (project_id))
  dataset_id = createDataset('190719_Hela_Ecoli_1to1_01 - tile 33')
  print('Created dataset: %s' % (dataset_id))
  interface_id = get_image_labeling_interface_id()
  ontology = {
      "tools": [
          {
              "color": "#FF5733",
              "tool": "rectangle",
              "name": "charge-1"
          },
          {
              "color": "#FFBD33",
              "tool": "rectangle",
              "name": "charge-2"
          },
          {
              "color": "#DBFF33",
              "tool": "rectangle",
              "name": "charge-3"
          },
          {
              "color": "#75FF33",
              "tool": "rectangle",
              "name": "charge-4"
          },
          {
              "color": "#33FFBD",
              "tool": "rectangle",
              "name": "charge-5"
          }
      ]
  }

  configure_interface_for_project(ontology, project_id, interface_id, org_id)
  completeSetupOfProject(project_id, dataset_id, interface_id)
  print('Attached dataset and interface to created project')

  prediction_model_id = create_prediction_model('PDA labelling model', 1)
  attach_prediction_model_to_project(prediction_model_id, project_id)

  print('Created and attached prediction model: %s' % (prediction_model_id))



  for row in file_labels:
    data_row_id = create_datarow(row['image_url'], row['external_id'], dataset_id)
    print('Created data row: %s' % (data_row_id))
    prediction_id = create_prediction(json.dumps(row['prediction_label']), prediction_model_id, project_id, data_row_id)
    print('Created prediction: %s' % (prediction_id))


  print('Rebuilt labeling queue since data was added')
  print('Go to https://app.labelbox.com/projects/%s/overview and click start labeling' % (project_id))
