pipeline {
	agent any
	stages {
		stage('Install virtualenv') {
			options {
				timeout(time: 1, unit: 'HOURS')
			}
			steps {
				sh 'python3 -m virtualenv venv -p python3'
				sh '. venv/bin/activate'
				sh 'echo . venv/bin/activate > activate_cmd'
				sh 'cat activate_cmd'
			}
			
		}
		stage('Install Requirements') {
			steps {
				sh '`cat activate_cmd` && python3 -m pip install -r requirements.txt'
			}
		}
		stage('Testing') {
			parallel {
				stage('Linter') {
					steps {
						sh '`cat activate_cmd` && python3 -m flake8 HPTuner/'
					}
				}
				stage('Tester') {
					steps {
						sh '`cat activate_cmd` && python3 -m pytest HPTuner --junit-xml report.xml' 
					}
					post {
						always {
							junit '**/report.xml'
						}
					}
				}	
			}
		}
		stage('Installer') {
			when {
				expression {
					return env.BRANCH_NAME == 'develop';
				}
			}
			steps {
				sh '`cat activate_cmd` && python3 setup.py bdist_wheel'
			}
			post {
				always {
					archiveArtifacts artifacts: 'dist/*.whl'
				}
			}
		}
	}
}

