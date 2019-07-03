import {Camera} from './camera';

describe('Camera', () => {
  it('should pass dummy test', () => {
    const camera = Camera.parse({
      id: 1,
      camera_url: 'test',
      name: 'test',
      status: 'active',
      url_stream: 'rtsp://test.test'
    });

    expect(camera).toBeDefined();
    expect(camera.id).toEqual(1);
  });
});
