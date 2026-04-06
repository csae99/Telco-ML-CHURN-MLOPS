pipeline {
    agent {
        docker {
            image 'python:3.11-slim'
        }
    }
    environment {
        IMAGE_NAME = "churn-api"
        TAG = "${BUILD_NUMBER}"
        REGISTRY = "docker.io/YOUR_USERNAME"   // change this
        FULL_IMAGE = "${REGISTRY}/${IMAGE_NAME}:${TAG}"
    }

    stages {

        stage('Checkout Code') {
            steps {
                git branch: 'churn-api-lite',
                    url: 'https://github.com/csae99/Telco-ML-CHURN-MLOPS.git'
            }
        }
        stage('Setup Python Environment') {
            steps {
                sh '''
                rm -rf venv
                python3 -m venv venv
                . venv/bin/activate
                python --version
                pip install --upgrade pip
                pip install -r requirements.txt
                '''
            }
        }
        stage('Prepare Processed_data') {
            steps {
                sh '''
                . venv/bin/activate
                python3 scripts/prepare_processed_data.py
                '''
            }
        }
        stage('Hyperparameter Tuning') {
            steps {
                sh '''
                . venv/bin/activate
                python3 scripts/test_pipeline_phase1_data_features.py
                python3 scripts/test_pipeline_phase2_modeling.py
                '''
            }
        }

        stage('Verify best_params.json') {
            steps {
                sh '''
                test -f best_params.json || (echo "best_params.json not found!" && exit 1)
                '''
            }
        }

        stage('Train Model Pipeline') {
            steps {
                sh '''
                set -x
                . venv/bin/activate
                python3 scripts/run_pipeline.py --input data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv --target Churn
                '''
            }
        }

        stage('Build Docker Image') {
            steps {
                sh '''
                docker build -t $FULL_IMAGE .
                '''
            }
        }

        // stage('Login to Docker Hub') {
        //     steps {
        //         withCredentials([usernamePassword(
        //             credentialsId: 'dockerhub-creds',
        //             usernameVariable: 'DOCKER_USER',
        //             passwordVariable: 'DOCKER_PASS'
        //         )]) {
        //             sh '''
        //             echo $DOCKER_PASS | docker login -u $DOCKER_USER --password-stdin
        //             '''
        //         }
        //     }
        // }

        // stage('Push Docker Image') {
        //     steps {
        //         sh '''
        //         docker push $FULL_IMAGE
        //         '''
        //     }
        // }

        stage('Deploy Container (Optional)') {
            steps {
                sh '''
                docker stop churn-container || true
                docker rm churn-container || true

                docker run -d \
                    -p 8000:8000 \
                    -p 7860:7860 \
                    --name churn-container \
                    $FULL_IMAGE
                '''
            }
        }
    }

    post {
        success {
            echo "✅ Pipeline completed successfully!"
        }
        failure {
            echo "❌ Pipeline failed!"
        }
    }
}
