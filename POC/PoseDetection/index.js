const webcamElement = document.getElementById('webcam');

async function app() {
  const detectorConfig = {modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING};
  const detector = await poseDetection.createDetector(poseDetection.SupportedModels.MoveNet, detectorConfig);
  const webcam = await tf.data.webcam(webcamElement);

  while (true) {

    const image = await webcam.capture();

    const poses = await detector.estimatePoses(image);
    console.log(poses[0].score);

    // Dispose the tensor to release the memory.
    image.dispose();

    await tf.nextFrame();
  }
}

app();
