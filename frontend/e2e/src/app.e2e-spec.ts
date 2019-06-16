import {AppPage} from './app.po';
import {by, element} from 'protractor';

describe('FaceAnalytics App', () => {
  let page: AppPage;

  beforeEach(() => {
    page = new AppPage();
  });

  it('should be displayed', () => {
    page.navigateTo();

    expect(element(by.css('app-root'))).toBeDefined();
  });
});
