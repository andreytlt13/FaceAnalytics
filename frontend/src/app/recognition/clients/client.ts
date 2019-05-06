export class Client {
  constructor(
    public id: string,
    public photo: string,
    public description: string,
    public rate?: number,
  ) {}

  static parse(json: any) {
    return new Client(json.id, json.photo, json.description, json.rate);
  }
}
